from keras.layers import Input
from keras import layers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2

def loadModel():
    img_input = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    y = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Convolutional Block 1
    print('BLOCK 1')
    x = Conv2D(64, 3, padding='same', strides=(2,2))(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(128, (1, 1), strides=(2,2))(y)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    y = Dropout(0.2)(x)

    # Convolutional Block 2
    print('BLOCK 2')
    x = Conv2D(128, 3, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, padding='same', strides=(2,2))(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(256, (1, 1), strides=(2,2))(y)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    y = Dropout(0.2)(x)

    # Convolutional Block 3
    print('BLOCK 3')
    x = Conv2D(256, 3, padding='same')(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, 3, padding='same', strides=(2,2))(y)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(512, (1, 1), strides=(2,2))(y)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    x = Dropout(0.2)(x)

    # classification block
    print('CLASSIFICATION BLOCK')
    x = AveragePooling2D((3,3))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    x = Dense(15, activation='softmax')(x)

    model = Model(img_input, x)

    return model