import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras.backend as K
import numpy as np
import tensorflow as tf
import config


def classification_loss(y_truth, y_pred):
    cls_mask = K.cast(y_truth >= 0, tf.float32)
    cross_entropy = K.binary_crossentropy(y_truth, y_pred) * cls_mask
    cross_entropy = K.sum(cross_entropy) / K.sum(cls_mask)
    return cross_entropy


def regression_loss(y_truth, y_pred):
    cls_label = y_truth[..., 4 * config.number_of_anchor:]
    reg_mask = K.cast(cls_label > 0, tf.float32)
    reg_non_mask=K.cast(cls_label <=0, tf.float32)
    num_anchors= (K.sum(reg_mask)+K.sum(reg_non_mask))/4
    y_truth_reg = y_truth[..., :4*config.number_of_anchor]
    x = y_truth_reg - y_pred
    x_abs = K.abs(x)
    x_bool = K.cast(K.less_equal(x_abs, config.epsilon), tf.float32)
    loss=config.lamda * K.sum(
        reg_mask * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / num_anchors
    return loss