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

from mrcnn.ROIAlignLayer import *
from mrcnn.MatterportDataFormatting import *



def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    with tf.variable_scope("fpn_classifier_graph", reuse=tf.AUTO_REUSE):
        # ROI Pooling
        pyramid_roi_align_class = PyramidROIAlign([pool_size, pool_size], name="roi_align_classifier")
        x = pyramid_roi_align_class.run([rois, image_meta] + feature_maps)

        x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, fc_layers_size, (pool_size, pool_size), padding="valid"), x,parallel_iterations=19, name="mrcnn_class_conv1")
        x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_class_bn1")
        x = tf.nn.relu(x)
        x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, fc_layers_size, (1, 1)), x, parallel_iterations=19,name="mrcnn_class_conv2")
        x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_class_bn2")
        x = tf.nn.relu(x)

        shared_function = lambda x: tf.squeeze(tf.squeeze(x, 3), 2)
        shared = shared_function(x)

        mrcnn_class_logits = tf.map_fn(lambda inputX: tf.layers.dense(inputX, int(num_classes)), shared, parallel_iterations=19,name="mrcnn_class_logits")
        mrcnn_probs = tf.map_fn(lambda inputX: tf.nn.softmax(inputX), mrcnn_class_logits, parallel_iterations=19)

        x = tf.map_fn(lambda inputX: tf.layers.dense(inputX, int(num_classes*4), activation='linear'), shared,parallel_iterations=19, name='mrcnn_bbox_fc')
        x = tf.expand_dims(x, 1)
        s = tf.shape(x, out_type=tf.int32)
        mrcnn_bbox = tf.reshape(x, (s[0], s[2], num_classes, 4), name="mrcnn_bbox")

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, [x,mrcnn_bbox, s]



def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):

    def map_fn_build_fpn_mask_graph(x, train_bn=False):

        x = tf.layers.conv2d(x, 256, (3, 3), padding="same")
        x = tf.layers.batch_normalization(x, training=train_bn)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 256, (3, 3), padding="same")
        x = tf.layers.batch_normalization(x, training=train_bn)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 256, (3, 3), padding="same")
        x = tf.layers.batch_normalization(x, training=train_bn)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, 256, (3, 3), padding="same")
        x = tf.layers.batch_normalization(x, training=train_bn)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x, 256, (2, 2), strides=2, activation="relu")
        x = tf.layers.conv2d(x, num_classes, (1, 1), strides=1, activation="sigmoid")

        return x


    with tf.variable_scope("build_fpn_mask_graph", reuse=tf.AUTO_REUSE):
        pyramid_roi_align_class = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")
        x = pyramid_roi_align_class.run([rois, image_meta] + feature_maps)

        testMapFnDebug = False
        if testMapFnDebug:
            x = tf.map_fn(lambda inputX: map_fn_build_fpn_mask_graph(inputX), x, parallel_iterations=19,name="build_fpn_mask_graph_map_fn")
            return x

        else:
            x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, 256, (3, 3), padding="same"), x, parallel_iterations=19,name="mrcnn_mask_conv1")
            x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_mask_bn1")
            x = tf.nn.relu(x)

            x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, 256, (3, 3), padding="same"), x, parallel_iterations=19,name="mrcnn_mask_conv2")
            x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_mask_bn2")
            x = tf.nn.relu(x)

            x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, 256, (3, 3), padding="same"), x, parallel_iterations=19,name="mrcnn_mask_conv3")
            x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_mask_bn3")
            x = tf.nn.relu(x)

            x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, 256, (3, 3), padding="same"), x, parallel_iterations=19,name="mrcnn_mask_conv4")
            x = tf.map_fn(lambda inputX: tf.layers.batch_normalization(inputX, training=train_bn), x, parallel_iterations=19,name="mrcnn_mask_bn4")
            x = tf.nn.relu(x)

            x = tf.map_fn(lambda inputX: tf.layers.conv2d_transpose(inputX, 256, (2, 2), strides=2, activation="relu"), x, parallel_iterations=19,name="mrcnn_mask_deconv")
            x = tf.map_fn(lambda inputX: tf.layers.conv2d(inputX, num_classes, (1, 1), strides=1, activation="sigmoid"), x, parallel_iterations=19,name="mrcnn_mask")

            return x
