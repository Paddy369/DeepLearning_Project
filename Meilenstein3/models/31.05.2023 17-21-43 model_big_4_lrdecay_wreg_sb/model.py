from keras.layers import Input
from keras import layers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2

def block(input_tensor, kernel_size, filters, strides=(1,1), skipping=False):
    filters1, filters2 = filters

    # x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = Conv2D(filters1, kernel_size, padding='same', strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (1, 1))(x)
    x = BatchNormalization()(x)

    if skipping:
        shortcut = Conv2D(filters2, (1, 1), strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut if skipping else input_tensor])
    x = Activation('relu')(x)
    return x

def loadModel():
    img_input = Input(shape=(256, 256, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = block(x, 3, [64, 128], (2,2), True)
    x = block(x, 3, [64, 128])

    x = block(x, 3, [128, 256], (2,2), True)
    x = block(x, 3, [128, 256])

    x = block(x, 3, [256, 512], (2,2), True)
    x = block(x, 3, [256, 512])

    x = AveragePooling2D((3,3))(x)
    x = Flatten() (x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001)) (x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001)) (x)
    x = Dropout(0.2) (x)
    x = Dense(15, activation='softmax') (x)

    model = Model(img_input, x)

    return model