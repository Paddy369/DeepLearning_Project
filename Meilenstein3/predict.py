import json
import tensorflow as tf
from datetime import datetime
import pandas as pd
from IPython.display import display

BASE_DIR = '../images'
names = ["Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"]
imageCount = [100, 100, 394, 177, 88, 306, 30, 57, 95, 184, 100, 237, 100, 164, 270]

# load the data
predict_batches = tf.keras.utils.image_dataset_from_directory (
    BASE_DIR + '/testing',
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
model = tf.keras.models.load_model('saved_models/model')

# model.summary()

# predict the results
predict = model.predict(predict_batches)
# get dominant class
predict = tf.argmax(predict, axis=1)

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

pd_object = pd.read_json('prediction.json', typ='series')
df = pd.DataFrame(pd_object)
display(df)

