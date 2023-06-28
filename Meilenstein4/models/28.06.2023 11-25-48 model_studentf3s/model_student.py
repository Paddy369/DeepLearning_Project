import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils

def loadModel():
    img_input = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (5, 5), strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), strides=(2, 2))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(128, (1, 1), strides=(2, 2))(x)
    x = Conv2D(128, (1, 1), strides=(2, 2))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (1, 1))(x)
    x = Conv2D(256, (1, 1))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((7, 7))(x)
    x = Flatten() (x)
    x = Dense(15, activation='softmax') (x)

    model = Model(img_input, x)

    return model