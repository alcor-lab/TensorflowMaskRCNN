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

from mrcnn.DataGenerator import *
from mrcnn.UtilityFunctions import *
from mrcnn.ResNetGraph import *
from mrcnn.ProposalLayer import *
from mrcnn.ROIAlignLayer import *
from mrcnn.DetectionTargetLayer import *
from mrcnn.DetectionLayer import *
from mrcnn.RegionProposalNetwork import *
from mrcnn.LossFunctions import *
from mrcnn.FeaturePyramidNetwork import *
from mrcnn.MatterportDataFormatting import *


class MaskRCNN():
    "Main Mask RCNN architecture class that calls and manages all the other blocks of the network"

    def __init__(self, mode, config, model_dir = ""):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']

        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.model_path = ""
        self.set_log_dir()
        self.model = self.create(mode=mode, config=config)

    def create(self, mode, config):

        with tf.variable_scope("MaskRCNN", reuse = tf.AUTO_REUSE):
            assert mode in ['training', 'inference']

            h, w = config.IMAGE_SHAPE[:2]
            if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
                raise Exception("Image size must be dividable by 2 at least 6 times "
                                "to avoid fractions when downscaling and upscaling."
                                "For example, use 256, 320, 384, 448, 512, ... etc. ")

            input_image = tf.placeholder(shape=[None, None, None, config.IMAGE_SHAPE[2]], name="input_image", dtype=tf.float32)
            input_image_meta = tf.placeholder(shape=[None, config.IMAGE_META_SIZE],name="input_image_meta", dtype=tf.float32)

            if mode == "training":
                input_rpn_match = tf.placeholder(shape=[None, None, 1], name="input_rpn_match", dtype=tf.int32)
                input_rpn_bbox = tf.placeholder(shape=[None, None, 4], name="input_rpn_bbox", dtype=tf.float32)

                input_gt_class_ids = tf.placeholder(shape=[None, None], name="input_gt_class_ids", dtype=tf.int32)

                input_gt_boxes = tf.placeholder(shape=[None, None, 4], name="input_gt_boxes", dtype=tf.float32)

                gt_boxes_function = lambda x : norm_boxes_graph(x, tf.shape(input_image)[1:3])
                gt_boxes = gt_boxes_function(input_gt_boxes)

                if config.USE_MINI_MASK:
                    input_gt_masks = tf.placeholder(shape=[None, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],name="input_gt_masks", dtype=bool)
                else:
                    input_gt_masks = tf.placeholder(shape=[None, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], name="input_gt_masks", dtype=bool)
            elif mode == "inference":
                input_anchors = tf.placeholder(shape=[None, None, 4], name="input_anchors", dtype=tf.float32)

            if callable(config.BACKBONE):
                _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                    train_bn=config.TRAIN_BN)
            else:
                _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                                 stage5=True, train_bn=config.TRAIN_BN)
            self.C_s = [C2,C3,C4,C5]

            with tf.variable_scope('Pyramid', reuse=tf.AUTO_REUSE):
                P5 = tf.layers.conv2d(C5, config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
                C4_conv = tf.layers.conv2d(C4, config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
                P5_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5)
                P4 = tf.add(P5_upsampled, C4_conv, name="fpn_p4add")

                C3_conv = tf.layers.conv2d(C3, config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
                P4_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4)
                P3 = tf.add(P4_upsampled, C3_conv, name="fpn_p3add")

                C2_conv = tf.layers.conv2d(C2, config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')
                P3_upsampled = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3)

                P2 = tf.add(P3_upsampled, C2_conv, name="fpn_p2add")


                P2 = tf.layers.conv2d(P2, config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")
                P3 = tf.layers.conv2d(P3, config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")
                P4 = tf.layers.conv2d(P4, config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")
                P5 = tf.layers.conv2d(P5, config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")

                P6 = tf.layers.max_pooling2d(P5, (1, 1), strides=(2, 2), name="fpn_p6")

            rpn_feature_maps = [P2, P3, P4, P5, P6]
            mrcnn_feature_maps = [P2, P3, P4, P5]


            if mode == "training":
                anchors = self.get_anchors(config.IMAGE_SHAPE)
                anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
                anchors = tf.keras.layers.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
            else:
                anchors = input_anchors

            layer_outputs = []
            with tf.name_scope('rpn_feature_maps'):
                for p in rpn_feature_maps:
                    layer_outputs.append(rpn_graph(p, len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE))

            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            outputs = list(zip(*layer_outputs))
            outputsAux = []
            for index in range(3):
                lista = list(outputs[index])
                elem = tf.keras.backend.concatenate(lista, axis=1)
                outputsAux.append(elem)

            rpn_class_logits, rpn_class, rpn_bbox = outputsAux

            proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
                else config.POST_NMS_ROIS_INFERENCE

            proposalLayerClass =  ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=config.RPN_NMS_THRESHOLD,
                name="ROI",
                config=config)
            rpn_rois = proposalLayerClass.run([rpn_class, rpn_bbox, anchors])

            if mode == "training":
                active_class_ids_function = lambda x: parse_image_meta_graph(x)["active_class_ids"]
                active_class_ids = active_class_ids_function(input_image_meta)
                if not config.USE_RPN_ROIS:
                    with tf.variable_scope("NOT_USE_RPN_ROIS", reuse=tf.AUTO_REUSE):
                        input_rois = tf.Variable(shape=[config.POST_NMS_ROIS_TRAINING, 4], name="input_roi", dtype=np.int32)
                        target_rois_function = lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3])
                        target_rois = target_rois_function(input_rois)
                else:
                    target_rois = rpn_rois

                detection_target_layer_class = DetectionTargetLayer(config, name="proposal_targets")
                targetVectorInput = [target_rois, input_gt_class_ids, gt_boxes, input_gt_masks]


                rois, target_class_ids, target_bbox, target_mask = detection_target_layer_class.graph(targetVectorInput)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, problem=\
                    fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN,
                                         fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

                mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

                output_rois_function = lambda x: x * 1
                output_rois = output_rois_function(rois)

                rpn_class_loss_function = lambda x: rpn_class_loss_graph(*x)
                rpn_bbox_loss_function  = lambda x: rpn_bbox_loss_graph(config, *x)
                class_loss_function     = lambda x: mrcnn_class_loss_graph(*x)
                bbox_loss_function      = lambda x: mrcnn_bbox_loss_graph(*x)
                mask_loss_function      = lambda x: mrcnn_mask_loss_graph(*x)
                rpn_class_loss = rpn_class_loss_function([input_rpn_match, rpn_class_logits])
                rpn_bbox_loss  = rpn_bbox_loss_function([input_rpn_bbox, input_rpn_match, rpn_bbox])
                class_loss     = class_loss_function([target_class_ids, mrcnn_class_logits, active_class_ids])
                bbox_loss      = bbox_loss_function([target_bbox, target_class_ids, mrcnn_bbox])
                mask_loss      = mask_loss_function([target_mask, target_class_ids, mrcnn_mask])

                inputs = [input_image, input_image_meta,
                          input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
                if not config.USE_RPN_ROIS:
                    inputs.append(input_rois)
                outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                           mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                           rpn_rois, output_rois,
                           rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]

            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, problem =\
                    fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN,
                                         fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
                detection_target_layer_class = DetectionLayer(mode, config, name="mrcnn_detection")

                targetVectorInput = [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
                detections = detection_target_layer_class.graph(targetVectorInput)
                detection_boxes_function = lambda x: x[..., :4]
                detection_boxes = detection_boxes_function(detections)

                mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  config.MASK_POOL_SIZE,
                                                  config.NUM_CLASSES,
                                                  train_bn=config.TRAIN_BN)

                inputs = [input_image, input_image_meta, input_anchors]
                outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]

            return inputs, outputs


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

#################################################
#           Model Managment FUctions            #
#################################################


    def saveModel(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, self.model_path)
        #save_path = saver.save(session, modelName)

    def loadModel(self):

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)


#####################################
#       Detection Fuctions          #
#####################################


    def timePerformanceTester(self, images, s, detectionIteration=30, verbose=0):

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        saver = tf.train.Saver()

        actual_input = [molded_images, image_metas, anchors]

        s.run(tf.global_variables_initializer())
        for i in range(detectionIteration):
            print("\n_________________________\nIteration:", i)
            start_aux = time.time()
            detections, _, _, mrcnn_mask, _, _, _ = \
                s.run(self.model[1][:8], feed_dict={i: d for i, d in zip(self.model[0], actual_input)})
            print("detection time is",time.time() - start_aux)


    def singleDetection(self, images, s, verbose=0):
        molded_images, image_metas, windows = self.mold_inputs(images)
        image_shape = molded_images[0].shape
        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        actual_input = [molded_images, image_metas, anchors]

        detections, _, _, mrcnn_mask, _, _, _ = \
            s.run(self.model[1][:8], feed_dict={i: d for i, d in zip(self.model[0], actual_input)})
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results


#####################################
#   Trianing FUction                #
#####################################

    def train(self, train_dataset, val_dataset, learning_rate, epochs=150):
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=False,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)


        inputs_train, _ = next(train_generator)
        inputs_val,   _ = next(val_generator)

        _, _, _, _, _, _, _, _, _,rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = self.model[1]
        rpn_class_loss_summary  = tf.summary.scalar("rpn_class_loss", rpn_class_loss)
        rpn_bbox_loss_summary   = tf.summary.scalar("rpn_bbox_loss", rpn_bbox_loss)
        class_loss_summary      = tf.summary.scalar("class_loss", class_loss)
        bbox_loss_summary       = tf.summary.scalar("bbox_loss", bbox_loss)
        mask_loss_summary       = tf.summary.scalar("mask_loss", mask_loss)

        summary_list = [rpn_class_loss_summary, rpn_bbox_loss_summary, class_loss_summary, bbox_loss_summary, mask_loss_summary]
        summary_merged = tf.summary.merge_all()
        with tf.name_scope('Loss_Trainig'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            one_by_one = False
            if one_by_one:
                train_op_rpn_class_loss = optimizer.minimize(loss=rpn_class_loss, global_step=tf.train.get_global_step())
                train_op_rpn_bbox_loss  = optimizer.minimize(loss=rpn_bbox_loss,  global_step=tf.train.get_global_step())
                train_op_class_loss     = optimizer.minimize(loss=class_loss,     global_step=tf.train.get_global_step())
                train_op_bbox_loss      = optimizer.minimize(loss=bbox_loss,      global_step=tf.train.get_global_step())
                train_op_mask_loss      = optimizer.minimize(loss=mask_loss,      global_step=tf.train.get_global_step())

                list_train_op = [train_op_rpn_class_loss, train_op_rpn_bbox_loss, train_op_class_loss,train_op_bbox_loss,train_op_mask_loss, summary_merged]

            else:
                losses = sum([rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss])
                train_op_losses = tf.contrib.layers.optimize_loss(losses, tf.train.get_global_step(), 0.001, optimizer)

                list_train_op = [train_op_losses, summary_merged]
                list_val_op = [losses, summary_merged]

        init_var = tf.global_variables_initializer()
        log_writer_train = tf.summary.FileWriter('logdir/train', tf.get_default_graph())
        log_writer_val = tf.summary.FileWriter('logdir/val', tf.get_default_graph())

        saver = tf.train.Saver()
        self.epoch = epochs
        print("Total Steps:", self.config.STEPS_PER_EPOCH)
        print("Total Epoches:", self.epoch,'\n')

        with tf.Session() as s:
            s.run(init_var)
            pbar_epoch = tqdm(total=(self.epoch), desc='Epoch', leave=False)
            for epoch_number in range(self.epoch):
                pbar_step = tqdm(total=(self.config.STEPS_PER_EPOCH*self.config.BATCH_SIZE), desc='Step', leave=False)
                for step_number in range(self.config.STEPS_PER_EPOCH):
                    actual_step = step_number*self.config.BATCH_SIZE + epoch_number * self.config.STEPS_PER_EPOCH * self.config.BATCH_SIZE

                    total_train_op = s.run(list_train_op, feed_dict={i: d for i, d in zip(self.model[0], inputs_train)})
                    log_writer_train.add_summary(total_train_op[-1],actual_step)

                    total_val_op = s.run(list_val_op, feed_dict={i: d for i, d in zip(self.model[0], inputs_val)})
                    log_writer_val.add_summary(total_val_op[-1],actual_step)

                    pbar_step.update(self.config.BATCH_SIZE)
                pbar_step.close()
                pbar_epoch.update(1)
                save_path = saver.save(s, self.model_path)
            log_writer_train.close()
            log_writer_val.close()
            pbar_epoch.close()
            save_path = saver.save(s, self.model_path)




#############################################################
#     Pre and Post Processing Matterport Mask RCNN FUctions #
#############################################################

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]



    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer




############################################################
#  Data Formatting
############################################################
