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


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    with tf.variable_scope("rpn_graph", reuse=tf.AUTO_REUSE):
        shared = tf.layers.conv2d(feature_map, 512, (3, 3), padding='same', activation='relu', strides=anchor_stride, name='rpn_conv_shared')
        x = tf.layers.conv2d(shared, 2 * anchors_per_location, (1, 1), padding='valid', activation='linear', name='rpn_class_raw')
        rpn_class_logits_function = lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2])
        rpn_class_logits = rpn_class_logits_function(x)

        rpn_probs = tf.nn.softmax(rpn_class_logits, name="rpn_class_xxx")

        x = tf.layers.conv2d(shared, anchors_per_location * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred')

        rpn_bbox_function = lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4])
        rpn_bbox = rpn_bbox_function(x)


        return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    input_feature_map = tf.placeholder(shape=[None, None, None, depth], name="input_rpn_feature_map", dtype=tf.float32)
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return input_feature_map, outputs

class build_rpn_model_class:

    def __init__(self,anchor_stride, anchors_per_location, depth):
        self.input_feature_map = tf.placeholder(shape=[None, None, None, depth], name="input_rpn_feature_map", dtype=tf.float32)


    def evalOutputs(self, input_feature_map_value):
        self.op = tf.assign(self.input_feature_map, input_feature_map_value, validate_shape=False)
        self.outputs = rpn_graph(self.op, anchors_per_location, anchor_stride)
        return self.outputs
