from keras.layers import Convolution2D, Input
from keras.models import Model
from tensorflow.keras import layers

# write a loadmodel with functional API based on resnet50

def loadModel():
    input = Input(shape=(256,256,3))
    print(input.shape)
    x = Convolution2D(8, (3, 3), activation='relu', padding='same')(input)
    x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2,2))(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2, seed=1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(15, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    return model
