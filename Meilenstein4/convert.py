import tensorflow as tf
import json
import tensorflow_model_optimization as tfmot
import numpy as np

# load the model settings
with open("config.json", 'r') as file:
    settings = json.load(file)

with open("compression_config.json", 'r') as file:
    compression_settings = json.load(file)

################### SETTINGS ###################

ID = settings["model_id"]                              
BASE_DIR = "../images/"                                 
image_size = settings["image_size"]                     
train_batch_size = settings["batch_size_train"]       
augmentation = settings["augmentation"]                
pruning_settings = compression_settings["pruning"]      
initial_sparsity = pruning_settings["initial_sparsity"]
final_sparsity = pruning_settings["final_sparsity"]          
epochs = pruning_settings["epochs"]

################################################

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

train_path = "train_aug" if augmentation else "testing"
train_batches = load_data(train_path, train_batch_size)

# load the model
model = tf.keras.models.load_model("saved_models/student_" + ID)

# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
end_step = np.ceil(len(train_batches)).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

model = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # optimize the model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # quantize the model
converter.target_spec.supported_types = [tf.float16]

# convert the model
tflite_model = converter.convert()

# save the model
with open('saved_models/student_%s.tflite' % ID, 'wb') as f:
    f.write(tflite_model)