import tensorflow as tf
import numpy as np
import keras
import keras_model as keras_modellib
import tensorflow_model as tensorflow_modellib
import coco


MODEL_DIR = "/home/alcor-lab/Desktop/km/convert/dfasdfasdfsfdfsadfasdf"

# PATH FOR KERAS MODEL
COCO_MODEL_PATH = "/home/alcor-lab/Desktop/km/convert/weight_converter/mask_rcnn_visiope_0185.h5"


class InferenceConfig(coco.CocoConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 15

config = InferenceConfig()
config.display()

model_keras = keras_modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model_keras.load_weights(COCO_MODEL_PATH, by_name=True)

saver = tf.train.Saver()
sess = keras.backend.get_session()
# SAVING KERAS WEIGHTS IN CKPT FORMAT
save_path = saver.save(sess, "/home/alcor-lab/Desktop/km/convert/weight_converter/coco_ktt/model.ckpt")



#=============================================#


tf_keras_model_path = "/home/alcor-lab/Desktop/km/convert/weight_converter/coco_ktt/model.ckpt" #NEWLY CONVERTED CKPT MODEL'S PATH
tensorflow_model_path = "/home/alcor-lab/Desktop/km/convert/weight_converter/empty/train_model.ckpt" #EMPTY MODEL WHICH WE WILL FILL WITH KERAS'S WEIGHTS
converted_tf_model = "/home/alcor-lab/Desktop/km/convert/weight_converter/final/model.ckpt" #FINAL MODEL WHICH WE NEED

class TrainConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 15

# CREATING NEW MODEL WHICH HAS OUR GRAPH STRUCTURE
config = TrainConfig()
model = tensorflow_modellib.MaskRCNN(mode="training", config=config)
model.saveModel(modelName=tensorflow_model_path)

#LOADING VARIABLES FROM OUR MODEL'S GRAPH
saver = tf.train.import_meta_graph(tensorflow_model_path+'.meta')
graph = tf.get_default_graph()
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])

# CHECKING WHICH VARIABLES HAVE IDENTICAL NAMES AND SAVING THEM IN A LIST
reader = tf.train.NewCheckpointReader(tf_keras_model_path)
param_map = reader.get_variable_to_shape_map()
lst_2 = []
for k, v in param_map.items():
    if 'Momentum' not in k and 'global_step' not in k:
        temp = np.prod(v)
        lst_2.append(k)

#LOADING VARIABLES FROM KERAS MODEL'S GRAPH
saver2 = tf.train.import_meta_graph(tf_keras_model_path+'.meta')
graph2 = tf.get_default_graph()
reuse_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict_2 = dict([(var.op.name, var) for var in reuse_vars2])
print("read all metas")

# PAIRING VARIABLES WITH DIFFERENT NAMES
pair_dict = {'MaskRCNN/fpn_classifier_graph/conv2d/bias': 'mrcnn_class_conv1/bias',
            'MaskRCNN/fpn_classifier_graph/conv2d_1/bias': 'mrcnn_class_conv2/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d/bias':'mrcnn_mask_conv1/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d_1/bias':'mrcnn_mask_conv2/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d_2/bias':'mrcnn_mask_conv3/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d_3/bias':'mrcnn_mask_conv4/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d_transpose/bias':'mrcnn_mask_deconv/bias',
            'MaskRCNN/build_fpn_mask_graph/conv2d_4/bias':'mrcnn_mask/bias',
            "MaskRCNN/fpn_classifier_graph/conv2d/kernel":"mrcnn_class_conv1/kernel",
            "MaskRCNN/fpn_classifier_graph/batch_normalization/gamma":"mrcnn_class_bn1/gamma",
            "MaskRCNN/fpn_classifier_graph/batch_normalization/beta":"mrcnn_class_bn1/beta",
            "MaskRCNN/fpn_classifier_graph/batch_normalization/moving_mean":"mrcnn_class_bn1/moving_mean",
            "MaskRCNN/fpn_classifier_graph/batch_normalization/moving_variance":"mrcnn_class_bn1/moving_variance",
            "MaskRCNN/fpn_classifier_graph/conv2d_1/kernel":"mrcnn_class_conv2/kernel",
            "MaskRCNN/fpn_classifier_graph/batch_normalization_1/gamma":"mrcnn_class_bn2/gamma",
            "MaskRCNN/fpn_classifier_graph/batch_normalization_1/beta":"mrcnn_class_bn2/beta",
            "MaskRCNN/fpn_classifier_graph/batch_normalization_1/moving_mean":"mrcnn_class_bn2/moving_mean",
            "MaskRCNN/fpn_classifier_graph/batch_normalization_1/moving_variance":"mrcnn_class_bn2/moving_variance",
            "MaskRCNN/fpn_classifier_graph/dense/kernel":"mrcnn_class_logits/kernel",
            "MaskRCNN/fpn_classifier_graph/dense/bias":"mrcnn_class_logits/bias",
            "MaskRCNN/fpn_classifier_graph/dense_1/kernel":"mrcnn_bbox_fc/kernel",
            "MaskRCNN/fpn_classifier_graph/dense_1/bias":"mrcnn_bbox_fc/bias",
            "MaskRCNN/build_fpn_mask_graph/conv2d/kernel":"mrcnn_mask_conv1/kernel",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization/gamma":"mrcnn_mask_bn1/gamma",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization/beta":"mrcnn_mask_bn1/beta",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization/moving_mean":"mrcnn_mask_bn1/moving_mean",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization/moving_variance":"mrcnn_mask_bn1/moving_variance",
            "MaskRCNN/build_fpn_mask_graph/conv2d_1/kernel":"mrcnn_mask_conv2/kernel",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_1/gamma":"mrcnn_mask_bn2/gamma",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_1/beta":"mrcnn_mask_bn2/beta",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_1/moving_mean":"mrcnn_mask_bn2/moving_mean",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_1/moving_variance":"mrcnn_mask_bn2/moving_variance",
            "MaskRCNN/build_fpn_mask_graph/conv2d_2/kernel":"mrcnn_mask_conv3/kernel",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_2/gamma":"mrcnn_mask_bn3/gamma",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_2/beta":"mrcnn_mask_bn3/beta",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_2/moving_mean":"mrcnn_mask_bn3/moving_mean",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_2/moving_variance":"mrcnn_mask_bn3/moving_variance",
            "MaskRCNN/build_fpn_mask_graph/conv2d_3/kernel":"mrcnn_mask_conv4/kernel",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_3/gamma":"mrcnn_mask_bn4/gamma",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_3/beta":"mrcnn_mask_bn4/beta",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_3/moving_mean":"mrcnn_mask_bn4/moving_mean",
            "MaskRCNN/build_fpn_mask_graph/batch_normalization_3/moving_variance":"mrcnn_mask_bn4/moving_variance",
            "MaskRCNN/build_fpn_mask_graph/conv2d_transpose/kernel":"mrcnn_mask_deconv/kernel",
            "MaskRCNN/build_fpn_mask_graph/conv2d_4/kernel":"mrcnn_mask/kernel"}

# CREATING DICTIONARIES WHICH WILL BE USED TO RESTORE ONLY NECESSARY VARIABLES FROM BOTH OUR AND KERAS' MODELS
to_change = {}
to_remain = {}
counter = 0
for k,v in reuse_vars_dict.items():
    counter+=1
    name = '/'.join(k.split('/',-1)[-2:])
    if name in lst_2:
        to_change[name] = v
    elif k in pair_dict:
        to_change[pair_dict[k]] = v
    else:
        to_remain[k] = v



# RESTORING NECESSARY VARIABLES
restore_saver = tf.train.Saver(to_remain)
restore_saver2 = tf.train.Saver(to_change)

# CREATING NEW MODEL WHICH HAS OUR GRAPH'S STRUCTURE AND WEIGHTS OF KERAS' MODEL
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    restore_saver.restore(sess, tensorflow_model_path)
    restore_saver2.restore(sess, tf_keras_model_path)

    saver.save(sess, converted_tf_model)
