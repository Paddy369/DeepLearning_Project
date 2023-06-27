import tensorflow as tf
import json
import tensorflow_model_optimization as tfmot
import numpy as np
from main import train_batches

# convert the model to a tflite model

# load the model settings
with open("config.json", 'r') as file:
    settings = json.load(file)

with open("compression_config.json", 'r') as file:
    compression_settings = json.load(file)

################### SETTINGS ###################
ID = settings["model_id"]  
train_batch_size = settings["batch_size_train"]       
pruning_settings = compression_settings["pruning"]      
initial_sparsity = pruning_settings["initial_sparsity"]
final_sparsity = pruning_settings["final_sparsity"]          
epochs = pruning_settings["epochs"]

# load the model
model = tf.keras.models.load_model("saved_models/" + ID)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # optimize the model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # quantize the model
converter.target_spec.supported_types = [tf.float16]

# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
num_images = train_batches.shape[0]
end_step = np.ceil(num_images / train_batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=0.80,
        begin_step=0,
        end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# convert the model
tflite_model = converter.convert()

# save the model
with open('saved_models/model_new4_2.tflite', 'wb') as f:
    f.write(tflite_model)