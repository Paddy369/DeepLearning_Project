import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, utils

# load the model settings
with open("config.json", 'r') as file:
    settings = json.load(file)

################### SETTINGS ###################

BASE_DIR = settings["base_dir"]                         # base directory
epochs = settings["epochs"]                             # number of epochs
names = settings["class_names"]                         # class names
train_batch_size = settings["batch_size_train"]         # batch size for training
val_batch_size = settings["batch_size_val"]             # batch size for training
test_batch_size = settings["batch_size_test"]           # batch size for training
image_size = settings["image_size"]                     # image size

################################################

# loads the data for a given path
def load_data(path, batch_size = 4):
    # read the images from a given directory and create a dataset with the subdirectories as labels
    data = tf.keras.utils.image_dataset_from_directory (
        BASE_DIR + path,                                # path to the data directory
        labels="inferred",                              # class labels are inferred from the subdirectory structure
        label_mode="int",                               # labels are returned as integers
        class_names=None,                               # names of the classes
        color_mode="rgb",                               # color images
        batch_size = batch_size,                        # number of images to retrieve at a time
        image_size=(image_size, image_size),            # images are resized to 128x128
        shuffle=True,                                   # shuffle the data
        seed=1,                                         # set the random seed for shuffling
        validation_split=None,                          # no data is used for validation
        subset=None,                                    # no data is used as a subset
        interpolation="bilinear",                       # interpolate images
        follow_links=False,                             # don't follow symbolic links
        crop_to_aspect_ratio=False                      # don't crop the images
    )
    return data

# load the data
train_batches = load_data("train", train_batch_size)    # training data is loaded in batches of 64
val_batches = load_data("validation", val_batch_size)   # validation data is loaded in batches of 8  
test_batches = load_data("testing", test_batch_size)    # testing data is loaded in batches of 8

# load the resnet model with imagenet weights and without the top layer
resnet = tf.keras.applications.resnet.ResNet50(
    include_top=False, 
    weights='imagenet', 
    input_shape=(image_size,image_size,3), 
    pooling=None
)

# freeze the layers
for layer in resnet.layers:
    layer.trainable = False

# create the model
model = Sequential()
# add the resnet model
model.add(resnet)

# add the top layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256, activation='softmax'))

# print the model summary
model.summary()

# compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

log_dir = "logs/log" + "_epochs_" + str(epochs)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_batches, validation_data=val_batches, callbacks=[tensorboard_callback], epochs=epochs)

model.evaluate(test_batches)

# # plot loss and acc
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='valid loss')
# plt.grid()
# plt.legend(fontsize=15)

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='train acc')
# plt.plot(history.history['val_accuracy'], label='valid acc')
# plt.grid()
# plt.legend(fontsize=15)
# plt.show()