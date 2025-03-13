import tensorflow as tf
input = tf.keras.layers.Input(shape=(), name='x', dtype='int64')
y = tf.keras.layers.ReLU(max_value=100, dtype='int64')(input)

