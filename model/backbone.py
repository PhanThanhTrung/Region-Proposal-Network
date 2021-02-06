import keras
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                     "releases/download/v0.1/" \
                     "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


def VGG16(pretrained_weight='imagenet', trainable=True):
    input_layer = Input(shape=(None, None, 3))

    conv_block1 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(input_layer)
    conv_block1 = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block1)
    conv_block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block1)

    conv_block2 = Conv2D(filters=128,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block1)
    conv_block2 = Conv2D(filters=128,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block2)
    conv_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block2)

    conv_block3 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block2)
    conv_block3 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block3)
    conv_block3 = Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block3)
    conv_block3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block3)

    conv_block4 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block3)
    conv_block4 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block4)
    conv_block4 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block4)
    conv_block4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block4)

    conv_block5 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block4)
    conv_block5 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block5)
    conv_block5 = Conv2D(filters=512,
                         kernel_size=(3, 3),
                         padding='same',
                         activation='relu')(conv_block5)
    conv_block5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_block5)

    if pretrained_weight == 'imagenet':
        VGG_Weights_path = keras.utils.get_file(
            pretrained_url.split('/')[-1], pretrained_url)
        Model(inputs=input_layer,
              outputs=conv_block5).load_weights(VGG_Weights_path)
    if trainable == False:
        for layer in Model(inputs=input_layer, outputs=conv_block5).layers[:]:
            layer.trainable = False

    return input_layer, conv_block1, conv_block2, conv_block3, conv_block4, conv_block5
