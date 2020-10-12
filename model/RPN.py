import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import config
import backbone
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input


def RPN():
    if config.backbone == 'VGG':
        backbone_blocks = backbone.VGG16()
    else:
        raise "Can't find valid backbone!"
    input_tensor = backbone_blocks[0]
    feature_maps = backbone_blocks[-1]
    conv_3x3 = Conv2D(filters=512,
                      kernel_size=(3, 3),
                      padding='same',
                      strides=(1, 1),
                      activation='relu',
                      kernel_initializer='normal',
                      name='rpn_3x3_conv')(feature_maps)
    reg_head = Conv2D(filters=4 * config.number_of_anchor,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      kernel_initializer='zero',
                      activation='linear',
                      name='rpn_reg_head')(conv_3x3)
    score_head = Conv2D(filters=config.number_of_anchor,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        kernel_initializer='uniform',
                        activation='sigmoid',
                        name='rpn_score_head')(conv_3x3)
    RPN_model = Model(inputs=[input_tensor], outputs=[reg_head, score_head])

    return RPN_model
