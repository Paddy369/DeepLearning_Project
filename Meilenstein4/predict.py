import json
import tensorflow as tf
from datetime import datetime
import pandas as pd
from IPython.display import display

BASE_DIR = '../images'
names = ["Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"]
imageCount = [100, 100, 379, 174, 87, 299, 28, 57, 93, 183, 93, 233, 98, 160, 263]

def printPretty(predict):
    for i in range(0, len(predict)):
        for j in range(0, len(names)):
            print("Image + " + str(i+1) + " - " + names[j] + ": " + "{:.4f}".format(predict[i][j]))

# function for creating a json file with the prediction results
def jsonify(predict): 
    results = {}
    # create dictionary for dominant classes and their count
    idx = 0
    # iterate over all classes
    for i in range(0, len(imageCount)):
        # create dictionary for dominant classes and their count for the current class
        results[names[i]] = {}
        # iterate over all images of the current class
        for j in range(0, imageCount[i]):
            # get the class name of the current image
            className = names[predict[idx]]
            # if the class name is already in the dictionary, increment the count 
            if className in results[names[i]]:
                results[names[i]][className] += 1
            # else add the class name to the dictionary
            else:
                results[names[i]][className] = 1
            idx += 1

    # Serializing json
    json_object = json.dumps(results, indent=4)
    
    # Writing to prediction.json
    with open("prediction.json", "w") as outfile:
        outfile.write(json_object)

# load the data
predict_batches = tf.keras.utils.image_dataset_from_directory (
    BASE_DIR + '/predict',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(256, 256),
    shuffle=False,
    seed=1,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# load the tensorflow model
model = tf.keras.models.load_model('saved_models/best_model_ms2')

# model.summary()

# predict the results
predict = model.predict(predict_batches)
printPretty(predict)
# get dominant class
predict = tf.argmax(predict, axis=1)
# persist the results in a json file
jsonify(predict)

