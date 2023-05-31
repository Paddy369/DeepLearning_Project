from tensorflow.keras import models, utils
import shutil
import os
import cv2
import numpy as np

# class names
names = ["Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"]

def classify(image, milestone, modelName):

    # copy the image into the demo directory
    shutil.copyfile("../frontend/public/img/" + image, "./demo/" + image)

    # load the image
    img = cv2.imread("./demo/" + image)

    # create batch from image
    batch = np.expand_dims(img, axis=0)

    # load the model
    try:
        # load the model
        model = models.load_model('../../' + milestone + '/saved_models/' + modelName)
        # predict the results
        predict = model.predict(batch)
        # map the predictions to the class names
        result = mapPredictions(predict)
    except:
        result = []

    # remove the image from the demo directory
    os.remove("./demo/" + image)

    # return the prediction
    return result

# maps the predictions to the class names
def mapPredictions(predictions):
    classes = {}
    for i in range(len(predictions[0])):
        classes[names[i]] = "{:.4f}".format(predictions[0][i])
    print(classes)
    return classes