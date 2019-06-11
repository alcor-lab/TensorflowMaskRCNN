import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf

import sys
import time
from mrcnn import utils
from tqdm import tqdm

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer:

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def run(self, inputs):
        with tf.variable_scope("ProposalLayer", reuse=tf.AUTO_REUSE):
            scores = inputs[0][:, :, 1]
            deltas = inputs[1]
            deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
            anchors = inputs[2]

            pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                             name="top_anchors").indices

            scores = tf.map_fn(lambda x: tf.gather(x[0],x[1]), (scores, ix),parallel_iterations=19, dtype=tf.float32)
            deltas = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (deltas, ix),parallel_iterations=19, dtype=tf.float32)
            pre_nms_anchors = tf.map_fn(lambda x: tf.gather(x[0],x[1]), (anchors, ix),parallel_iterations=19, dtype=tf.float32, name="pre_nms_anchors")
            boxes = tf.map_fn(lambda x: apply_box_deltas_graph(x[0], x[1]), (pre_nms_anchors, deltas),parallel_iterations=19, dtype=tf.float32, name="refined_anchors")

            window = np.array([0, 0, 1, 1], dtype=np.float32)
            boxes = tf.map_fn(lambda x: clip_boxes_graph(x, window), (boxes),parallel_iterations=19, dtype=tf.float32, name="refined_anchors_clipped")

            # Non-max suppression
            def nms(boxes, scores):
                indices = tf.image.non_max_suppression(
                    boxes, scores, self.proposal_count,
                    self.nms_threshold, name="rpn_non_max_suppression")
                proposals = tf.gather(boxes, indices)
                padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
                proposals = tf.pad(proposals, [(0, padding), (0, 0)])
                return proposals

            proposals = tf.map_fn(lambda x: nms(x[0], x[1]), (boxes, scores),parallel_iterations=19, dtype=tf.float32)
            return proposals
