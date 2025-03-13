# https://colab.research.google.com/drive/14KKrdiBg8FT2cdUC5pjqJHi5S3BkTOHi?usp=sharing
# The above colab will run fine, but the same code on Mac with the said config has issue.
# Copying the code here for quick reference.

import tensorflow as tf    
import tensorflow.keras
import tensorflow as tf
import platform
import sys
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model

# Print versions:
print(f"Python {sys.version}")
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Setup input
import numpy as np
X_check = np.array([[1, 0, 0]])

# Setup autoencoder model
input_layer = Input(shape=(X_check.shape[1]))
bottleneck = Dense(2, activation='relu', name='bottleneck')(input_layer)
output = Dense(X_check.shape[1], activation='sigmoid', name='output')(bottleneck)
autoencoder = Model(input_layer, output)

# Set encoder layer weights to all negative.
layer = autoencoder.layers[1]
weights = np.array([[-1, -1],[-1, -1], [-1, -1]])
biases = np.array([0, 0])
layer.set_weights([weights, biases])

# create encoder model.
encoder = Model(input_layer, bottleneck)

# create decoder model.
decoder_input = Input(shape=(2,), name='decoder_input')
decoder_layer = autoencoder.layers[-1]
decoder = Model(decoder_input, decoder_layer(decoder_input))

# Run auto-encoder, with [1, 0, 0], since encoder has all negative weights,
# and has 'relu' activation o/p of enocder should all be zeros. And that being
# the input of next sigmod we should get output [0.5, 0.5, 0.5]
output_data = autoencoder.predict(X_check)
print(output_data)

