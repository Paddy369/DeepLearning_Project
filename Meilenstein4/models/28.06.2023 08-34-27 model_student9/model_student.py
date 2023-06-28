from tensorflow.keras import Sequential, layers

def loadModel():
    model = Sequential()

    #add the layers
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

    #flatten the output
    model.add(layers.Flatten()) 
    
    #add the dense layers
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dropout(0.5, seed=1))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(15, activation='softmax'))    
    return model