import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(1)(inp)

model = tf.keras.Model(inp, out)

model.predict(
    np.zeros((32, 1)),
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="test")],
)

