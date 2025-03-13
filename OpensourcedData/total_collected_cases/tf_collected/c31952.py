import numpy as np
import tensorflow as tf

x = tf.cast(np.random.randn(100, 100), tf.float32)
z = tf.cast(np.random.randn(1, 100), tf.float32)

layer = tf.keras.layers.Dense(100)

@tf.function  # <- removing this and the code works fine
def fun(x, layer):
    y = layer(x)
    return y

with tf.GradientTape() as tape:
    y = fun(x, layer)
    y = tf.gather(y, [0])  # if we put this line inside the function it works fine
    loss = tf.norm(y - z)

grads = tape.gradient(loss, layer.trainable_variables)

