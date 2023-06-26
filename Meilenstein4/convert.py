import tensorflow as tf

# convert the model to a tflite model

# load the model
model = tf.keras.models.load_model("saved_models/model_new4_1")

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # optimize the model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # quantize the model
converter.target_spec.supported_types = [tf.float16]

# convert the model
tflite_model = converter.convert()

# save the model
with open('saved_models/model_new4_2.tflite', 'wb') as f:
    f.write(tflite_model)