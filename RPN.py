import numpy as np
import keras
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from backbone import VGG16

k = 9


def model():
    blocks = VGG16(pretrained_weight='imagenet')
    for conv_block_layer in Model(inputs=blocks[0], outputs=blocks[-1]).layers[:]:
        conv_block_layer.trainable = False
    conv3x3 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu')(blocks[-1])
    reg_head = Conv2D(filters=k * 4,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='linear',
                      kernel_initializer='uniform')(conv3x3)
    cls_head = Conv2D(filters=k * 1,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='sigmoid',
                      kernel_initializer='uniform')(conv3x3)
    model = Model(inputs=blocks[0], outputs=[reg_head, cls_head])
    return model


model = model()
model.summary()