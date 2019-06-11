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
from mrcnn.MatterportDataFormatting import *


def overlaps_graph(boxes1, boxes2):

    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    with tf.variable_scope("detection_targets_graph", reuse=tf.AUTO_REUSE):

        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")

        gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name="trim_gt_masks")

        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        overlaps = overlaps_graph(proposals, gt_boxes)

        crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]


        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= config.BBOX_STD_DEV

        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        boxes = positive_rois
        if config.USE_MINI_MASK:
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids,
                                         config.MASK_SHAPE)
        masks = tf.squeeze(masks, axis=3)
        masks = tf.round(masks)

        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks


class detection_targets_graph_class:
    def __init__(self, c):
        self.config = c

    def run(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        with tf.variable_scope("detection_targets_graph", reuse=tf.AUTO_REUSE):

            # Assertions
            asserts = [
                tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                          name="roi_assertion"),
            ]
            with tf.control_dependencies(asserts):
                proposals = tf.identity(proposals)

            proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")

            gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
            gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                           name="trim_gt_class_ids")
            gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                                 name="trim_gt_masks")

            crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
            non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
            crowd_boxes = tf.gather(gt_boxes, crowd_ix)
            crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
            gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
            gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
            gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

            overlaps = overlaps_graph(proposals, gt_boxes)

            crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
            crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
            no_crowd_bool = (crowd_iou_max < 0.001)

            roi_iou_max = tf.reduce_max(overlaps, axis=1)
            positive_roi_bool = (roi_iou_max >= 0.5)
            positive_indices = tf.where(positive_roi_bool)[:, 0]
            negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]


            positive_count = int(self.config.TRAIN_ROIS_PER_IMAGE *
                                 self.config.ROI_POSITIVE_RATIO)
            positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
            positive_count = tf.shape(positive_indices)[0]
            r = 1.0 / self.config.ROI_POSITIVE_RATIO
            negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
            negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
            positive_rois = tf.gather(proposals, positive_indices)
            negative_rois = tf.gather(proposals, negative_indices)

            # Assign positive ROIs to GT boxes.
            positive_overlaps = tf.gather(overlaps, positive_indices)
            roi_gt_box_assignment = tf.cond(
                tf.greater(tf.shape(positive_overlaps)[1], 0),
                true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
                false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
            )
            roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
            roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

            deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
            deltas /= self.config.BBOX_STD_DEV

            transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
            roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

            boxes = positive_rois
            if self.config.USE_MINI_MASK:
                y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
                gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
                gt_h = gt_y2 - gt_y1
                gt_w = gt_x2 - gt_x1
                y1 = (y1 - gt_y1) / gt_h
                x1 = (x1 - gt_x1) / gt_w
                y2 = (y2 - gt_y1) / gt_h
                x2 = (x2 - gt_x1) / gt_w
                boxes = tf.concat([y1, x1, y2, x2], 1)
            box_ids = tf.range(0, tf.shape(roi_masks)[0])
            masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                             box_ids,
                                             self.config.MASK_SHAPE)
            masks = tf.squeeze(masks, axis=3)
            masks = tf.round(masks)

            rois = tf.concat([positive_rois, negative_rois], axis=0)
            N = tf.shape(negative_rois)[0]
            P = tf.maximum(self.config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
            rois = tf.pad(rois, [(0, P), (0, 0)])
            roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
            roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
            deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
            masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

            return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer():

    def __init__(self, config, **kwargs):
        self.config = config

    def graph(self, inputs):
        with tf.variable_scope("DetectionTargetLayer", reuse=tf.AUTO_REUSE):
            proposals = inputs[0]
            gt_class_ids = inputs[1]
            gt_boxes = inputs[2]
            gt_masks = inputs[3]

            names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
            outputs = tf.map_fn(lambda x: detection_targets_graph(x[0], x[1], x[2], x[3], self.config),(proposals, gt_class_ids, gt_boxes, gt_masks),parallel_iterations=19, dtype=tf.float32)

            return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
