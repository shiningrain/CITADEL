import tensorflow as tf
from tensorflow import keras

inp = keras.Input(shape=(1))
x = tf.cast(inp, dtype=tf.float64)
a = tf.constant(1.0, dtype=tf.float64)
x = tf.maximum(a, x)
out = tf.cast(x, dtype=tf.float32)

model = keras.models.Model(inp, out)
model.summary()

model.save('dummy_model')
del model

loaded_model = tf.keras.models.load_model('dummy_model')

