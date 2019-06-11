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


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):

    with tf.variable_scope('identityResNet_stage' + str(stage) + '_block'+ str(block), reuse=tf.AUTO_REUSE):
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = tf.layers.conv2d(input_tensor, nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2a')
        x = tf.nn.relu(x, name='relu_2a')
        x = tf.layers.conv2d(x, nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2b')
        x = tf.nn.relu(x, name='relu_2b')
        x = tf.layers.conv2d(x, nb_filter3, (1, 1), name=conv_name_base +'2c', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2c')

        x = tf.add(x, input_tensor)
        x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')

        return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):

    with tf.variable_scope('convResNet_stage' + str(stage) + '_block'+ str(block), reuse=tf.AUTO_REUSE):
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = tf.layers.conv2d(input_tensor, nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2a')
        x = tf.nn.relu(x, name='relu_2a')
        x = tf.layers.conv2d(x, nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2b')
        x = tf.nn.relu(x, name='relu_2b')
        x = tf.layers.conv2d(x, nb_filter3, (1, 1), name=conv_name_base +'2c', use_bias=use_bias)
        x = tf.layers.batch_normalization(x, training=train_bn, name=bn_name_base + '2c')

        shortcut = tf.layers.conv2d(input_tensor, nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)
        shortcut = tf.layers.batch_normalization(shortcut, training=train_bn, name=bn_name_base + '1')

        x = tf.add(x, shortcut)
        x = tf.nn.relu(x, name='res' + str(stage) + block + '_out')
        return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    with tf.variable_scope('resnet_graph', reuse=tf.AUTO_REUSE) as scope:
        #scope.reuse_variables()
        assert architecture in ["resnet50", "resnet101"]

        # Stage 1
        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        x = tf.pad(input_image, paddings, "SYMMETRIC")
        x = tf.layers.conv2d(x, 64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)
        x = tf.layers.batch_normalization(x, training=train_bn,name='bn_conv1')
        x = tf.nn.relu(x, name='relu')
        C1 = x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding="same", name='C1')

        # Stage 2
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
        C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

        # Stage 3
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
        C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
        # Stage 4
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
        block_count = {"resnet50": 5, "resnet101": 22}[architecture]
        for i in range(block_count):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        C4 = x
        # Stage 5
        if stage5:
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
            C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
        else:
            C5 = None
        return [C1, C2, C3, C4, C5]
