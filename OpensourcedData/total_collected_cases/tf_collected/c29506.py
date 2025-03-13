# Imports
import numpy as np
import tensorflow as tf
tf.executing_eagerly()
print('TensorFlow version: ' + str(tf.__version__))

# Print checks
from tensorflow.python.eager import context
print('Executing eagerly? : ' + str(context.executing_eagerly()))
print('Number of GPUs: ' + str(context.num_gpus()))

# Generate random data
X = np.random.rand(6720,700,3)
y = X[:,1,1]
print('Shapes: ', X.shape, y.shape)

# Define toy network
input_shape = X.shape[2]
rnn_state_size = 1
timesteps = X.shape[1]

inputs = tf.keras.layers.Input(shape=[timesteps, input_shape], dtype=np.float32)
output = tf.keras.layers.LSTM(rnn_state_size)(inputs)
model = tf.keras.Model(inputs, output)
model.compile('rmsprop', 'mse')
print(model.summary())

# Fit
model.fit(X,y)

