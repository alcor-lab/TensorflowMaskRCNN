import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
import tensorflow as tf

class Mask:

    def __init__(self, img_per_gpu=1, gpu_percentage=0.8):
        config_s = tf.ConfigProto()
        config_s.gpu_options.per_process_gpu_memory_fraction = gpu_percentage
        self.s = tf.Session(config=config_s)
        self.class_list  = ['BG','spraybottle', 'screwdriver', 'torch', 'cloth', 'cutter', 'pliers', 'brush', 'torch_handle', 'guard', 'ladder', 'closed_ladder', 'guard-support', 'robot', 'technician', 'diverter' ]

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = img_per_gpu
            NUM_CLASSES = len(self.class_list)


        self.latest_ckp = tf.train.latest_checkpoint('checkpoint')
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config)
        self.saver = tf.train.Saver()
        self.s.run(tf.global_variables_initializer())
        self.saver.restore(self.s, self.latest_ckp)


    def singleImageDetection(self, image, visualize_option=False, print_option=True):
        results = self.model.singleDetection(image, self.s, verbose=1)
        result  = results[0]

        if visualize_option:
            visualize.display_instances(image[0], result['rois'], result['masks'], result['class_ids'], self.class_list, result['scores'])

        if print_option:
            for i in range(len(r['class_ids'])):
                print(self.class_list[result['class_ids'][i]] + '; ' + str(result['scores'][i]))

        return result

    def timeTester(self, image):
        result = self.model.timePerformanceTester(image, self.s, verbose=1)
