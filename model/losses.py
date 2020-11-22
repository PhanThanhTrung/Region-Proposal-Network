import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras.backend as K
import numpy as np
import tensorflow as tf
import config


def classification_loss(y_truth, y_pred):
    mask = K.cast(y_truth[y_truth >= 0], tf.float32)
    cross_entropy = K.binary_crossentropy(y_truth, y_pred) * mask
    cross_entropy = cross_entropy / K.sum(mask)
    return cross_entropy


def regression_loss(y_truth, y_pred):
    cls_label = y_truth[..., -1]
    mask = K.cast(cls_label[cls_label > 0], tf.float32)
    y_truth_reg = y_truth[..., :4]
    x = y_truth_reg - y_pred
    x_abs = K.abs(x)
    x_bool = K.cast(K.less_equal(x_abs, config.epsilon), tf.float32)
    return config.lamda * K.sum(
        mask * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / (feature_map_height * feature_map_width *
                                   number_of_anchor)
