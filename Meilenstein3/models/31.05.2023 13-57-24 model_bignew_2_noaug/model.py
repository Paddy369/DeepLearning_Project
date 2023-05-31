from keras.layers import Input
from keras import layers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Model

def block(input_tensor, kernel_size, filters, strides=(1,1), skipping=False):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    if skipping:
        shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut if skipping else input_tensor])
    x = Activation('relu')(x)
    return x

def loadModel():
    img_input = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = block(x, 3, [256, 256, 1024], (2,2), True)
    x = block(x, 3, [256, 256, 1024])

    x = block(x, 3, [512, 512, 2048], (2,2), True)
    x = block(x, 3, [512, 512, 2048])

    x = AveragePooling2D((3,3))(x)
    x = Flatten() (x)
    x = Dense(128, activation='relu') (x)
    x = Dropout(0.5) (x)
    x = Dense(15, activation='softmax') (x)

    model = Model(img_input, x)

    return model