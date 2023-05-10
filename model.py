import json
import shutil
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, Sequential, optimizers, losses, callbacks

# load the model settings
with open("config.json", 'r') as file:
    settings = json.load(file)

################### SETTINGS ###################

BASE_DIR = "images/"                                    # base directory
epochs = settings["epochs"]                             # number of epochs
train_batch_size = settings["batch_size_train"]         # batch size for training
val_batch_size = settings["batch_size_val"]             # batch size for training
test_batch_size = settings["batch_size_test"]           # batch size for training
image_size = settings["image_size"]                     # image size
augmentation = settings["augmentation"]                 # is augmentation enabled
weight_decay = settings["weight_decay"]                 # is weight decay enabled
learning_rate_decay = settings["learning_rate_decay"]   # is learning rate decay enabled
layer_settings = settings["layers"]                     # layer configurations

################################################

# loads the data for a given path
def load_data(path, batch_size = 4):
    # read the images from a given directory and create a dataset with the subdirectories as labels
    data = tf.keras.utils.image_dataset_from_directory (
        BASE_DIR + path,                                # path to the data directory
        labels="inferred",                              # class labels are inferred from the subdirectory structure
        label_mode="int",                               # labels are returned as integers
        class_names= ["Beetle", "Butterfly", "Cat",     # names of the classes
                      "Cow", "Dog", "Elephant", 
                      "Gorilla", "Hippo", "Lizard", 
                      "Monkey", "Mouse", "Panda", 
                      "Spider", "Tiger", "Zebra"],      
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
train_path = "train_aug" if augmentation else "train"
train_batches = load_data(train_path, train_batch_size) # training data is loaded in batches of 64
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

# load the layer configurations and add them to the model as the top layers
for ls in layer_settings:
    if ls["type"] == "flatten":
        model.add(layers.Flatten())
    elif ls["type"] == "batch_normalization":
        model.add(layers.BatchNormalization())
    elif ls["type"] == "dense":
        model.add(layers.Dense(ls["size"], activation=ls["activation"]))
    elif ls["type"] == "dropout":
        model.add(layers.Dropout(ls["value"]))

# print the model summary
model.summary()

# loop for testing out different learning rates
for i in range(0, 5):
    # calculate the learning rate
    learning_rate = 10**(-i)

    # compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    # path to the log directory
    now = datetime.now()
    log_dir = "logs/" + now.strftime("%d.%m.%Y %H-%M-%S") + " lr_" + str(learning_rate)

    # copy the config file to the log directory
    os.makedirs("./" + log_dir, exist_ok=True)
    shutil.copyfile("config.json", "./" + log_dir + "/config.json")

    # create a callback to log the data for tensorboard
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train the model
    history = model.fit(train_batches, validation_data=val_batches, callbacks=[tensorboard_callback], epochs=epochs)

    # evaluate the model
    results = model.evaluate(test_batches, verbose=2)

    # save the test results
    f = open(log_dir + "/test_results.txt", "a")
    f.write("Loss: " + str(results[0]) + ", Accuracy: " + str(results[1]))
    f.close()