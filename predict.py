import json
import shutil
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, Sequential, optimizers, losses, callbacks

BASE_DIR = 'images'
names = ["Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"]

# load the data
predict_batches = tf.keras.utils.image_dataset_from_directory (
    BASE_DIR + '/predict',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=1,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# load the tensorflow model
model = tf.keras.models.load_model('model')

model.summary()

predict = model.predict(predict_batches)
predict = tf.argmax(predict, axis=1)
print(predict)