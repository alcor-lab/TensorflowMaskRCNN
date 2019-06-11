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



def smooth_l1_loss(y_true, y_pred):
    with tf.variable_scope("smooth_l1_loss", reuse=tf.AUTO_REUSE):
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.math.less(diff, 1.0), tf.float32)
        loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

        return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    with tf.variable_scope("rpn_class_loss_graph", reuse=tf.AUTO_REUSE):
        rpn_match = tf.squeeze(rpn_match, -1)
        anchor_class = tf.cast(tf.math.equal(rpn_match, 1), tf.int32)

        indices = tf.where(tf.math.not_equal(rpn_match, 0))

        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class, logits=rpn_class_logits)
        f1 = lambda: tf.reduce_mean(loss)
        f2 = lambda: tf.constant(0.0)
        loss = tf.case([(tf.math.greater(tf.size(loss),0), f1)], default=f2)

        return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    with tf.variable_scope("rpn_bbox_loss_graph", reuse=tf.AUTO_REUSE):
        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))

        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        batch_counts = tf.reduce_sum(tf.cast(tf.math.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

        loss = smooth_l1_loss(target_bbox, rpn_bbox)

        f1 = lambda: tf.reduce_mean(loss)
        f2 = lambda: tf.constant(0.0)
        loss = tf.case([(tf.math.greater(tf.size(loss),0), f1)], default=f2)
        return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    with tf.variable_scope("mrcnn_class_loss_graph", reuse=tf.AUTO_REUSE):

        target_class_ids = tf.cast(target_class_ids, 'int64')

        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)
        pred_class_logits = tf.transpose(pred_class_logits, [1,0,2])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)
        loss = tf.transpose(loss, [1,0])
        loss = loss * pred_active
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    with tf.variable_scope("mrcnn_bbox_loss_graph", reuse=tf.AUTO_REUSE):

        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox, out_type=tf.int32)[2], 4))

        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        f1 = lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)
        f2 = lambda: tf.constant(0.0)
        loss = tf.case([(tf.math.greater(tf.size(target_bbox),0), f1)], default=f2)
        loss = tf.reduce_mean(loss)
        return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    with tf.variable_scope("mrcnn_mask_loss_graph", reuse=tf.AUTO_REUSE):
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks,(-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

        positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ix), tf.int64)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)

        f1 = lambda: tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        f2 = lambda: tf.constant(0.0)
        loss = tf.case([(tf.math.greater(tf.size(y_true),0), f1)], default=f2)
        loss = tf.reduce_mean(loss)
        return loss
