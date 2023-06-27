import json
import shutil
import os
import tensorflow as tf
from model_student import loadModel
from datetime import datetime
from tensorflow.keras import optimizers, losses, callbacks, mixed_precision, metrics
from tensorflow.data.experimental import AUTOTUNE
from distiller import Distiller

# load the model settings
with open("config.json", 'r') as file:
    settings = json.load(file)

################### SETTINGS ###################

ID = settings["model_id"]                               # id of the model
BASE_DIR = "../images/"                                 # base directory
epochs = settings["epochs"]                             # number of epochs
train_batch_size = settings["batch_size_train"]         # batch size for training
val_batch_size = settings["batch_size_val"]             # batch size for training
test_batch_size = settings["batch_size_test"]           # batch size for training
predict_batch_size = settings["batch_size_predict"]     # batch size for training
image_size = settings["image_size"]                     # image size
augmentation = settings["augmentation"]                 # is augmentation enabled

################################################

# loads the data for a given path
def load_data(path, batch_size, label_mode="int"):
    # read the images from a given directory and create a dataset with the subdirectories as labels
    return tf.keras.utils.image_dataset_from_directory (
        BASE_DIR + path,                                # path to the data directory
        labels="inferred",                              # class labels are inferred from the subdirectory structure
        label_mode=label_mode,                          # labels are returned as integers
        class_names= ["Beetle", "Butterfly", "Cat",     # names of the classes
                      "Cow", "Dog", "Elephant", 
                      "Gorilla", "Hippo", "Lizard", 
                      "Monkey", "Mouse", "Panda", 
                      "Spider", "Tiger", "Zebra"],      
        color_mode="rgb",                               # color images
        batch_size = batch_size,                        # number of images to retrieve at a time
        image_size=(image_size, image_size),            # images are resized to 256x256
        shuffle=True,                                   # shuffle the data
        seed=1,                                         # set the random seed for shuffling
        validation_split=None,                          # no data is used for validation
        subset=None,                                    # no data is used as a subset
        interpolation="bilinear",                       # interpolate images
        follow_links=False,                             # don't follow symbolic links
        crop_to_aspect_ratio=False                      # don't crop the images
    )

# load the data
train_path = "train_aug" if augmentation else "train"
train_batches = load_data(train_path, train_batch_size) # training data is loaded in batches
val_batches = load_data("validation", val_batch_size)   # validation data is loaded in batches
test_batches = load_data("testing", test_batch_size)    # testing data is loaded in batches

# load the teacher and the student model
teacher_model = tf.keras.models.load_model("saved_models/model_ms2")
teacher_model.build(input_shape=(None, image_size, image_size, 3))
student_model = tf.keras.models.load_model("saved_models/model_student3", compile=False)
student_model.build(input_shape=(None, image_size, image_size, 3))

# print the model summary
# student_model.summary()

# initial learning rate
lr = settings["learning_rate"]["initial_lr"]

# path to the log directory and the model directory
now = datetime.now()
log_dir = "logs/" + now.strftime("%d.%m.%Y %H-%M-%S ") + str(ID)        # for our tensorboard logs and the related model configuration
model_dir = "models/" + now.strftime("%d.%m.%Y %H-%M-%S ") + str(ID)     # for transparancy reasons (wird mit ins repo eingecheckt)

if settings["learning_rate"]["decay"] == True:
    lr = optimizers.schedules.ExponentialDecay(lr, 
        decay_steps=settings["learning_rate"]["steps"], 
        decay_rate=settings["learning_rate"]["decay_rate"], 
        staircase=False, name=None)

# compile the model using the distiller
distiller = Distiller(teacher=teacher_model, student=student_model)
distiller.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    metrics=[metrics.SparseCategoricalAccuracy()],
    student_loss_fn=losses.SparseCategoricalCrossentropy(),
    distillation_loss_fn=losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# copy the config file and the current model configuration to the log directory
os.makedirs("./" + log_dir, exist_ok=True)
shutil.copyfile("config.json", "./" + log_dir + "/config.json")
shutil.copyfile("model_student.py", "./" + log_dir + "/model_student.py")
# copy the config file and the current model configuration to the model directory
os.makedirs("./" + model_dir, exist_ok=True)
shutil.copyfile("config.json", "./" + model_dir + "/config.json")
shutil.copyfile("model_student.py", "./" + model_dir + "/model_student.py")
# create a directory for the saved models
os.makedirs("./saved_models", exist_ok=True)

# create a callback to log the data for tensorboard
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
early_stopping_callback = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10)

def get_callbacks():
  return [
    callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10),
    callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True),
  ]

# train the model
history = distiller.fit(train_batches, validation_data=val_batches, callbacks=get_callbacks(), epochs=epochs)

# evaluate the model
results = distiller.evaluate(test_batches)

# save the test results
f = open(log_dir + "/test_results.txt", "a")
f.write("Accuracy: " + str(results[0]) + ", Student Loss: " + str(results[1]))
f.close()

# save the model
student_model.save("saved_models/" + ID)
