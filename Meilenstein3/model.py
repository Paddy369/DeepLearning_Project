from tensorflow.keras import Sequential, layers

def loadModel():
    model = Sequential()

    # Block 1
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))

    # Classification block
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(15, activation='softmax'))

    model.build((None, 256, 256, 3))

    return model