import tensorflow as tf
import numpy as np

filters, kernel_size, strides, padding = 3, [2, 2], 2, 'valid'
data = np.random.rand(1, 1, 1, 1)
layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
print(layer(data).shape)

