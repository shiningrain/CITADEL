import tensorflow as tf
import tensorflow_model_optimization as tfmot

i = tf.keras.layers.Input(shape=(24, 24, 3))
x = tf.keras.layers.Conv2D(10, kernel_size=1, activation='tanh')(i)
model = tf.keras.Model(inputs=i, outputs=x)

quant_aware_model = tfmot.quantization.keras.quantize_model(model)


**Case II:** Even worse - when specifying the activation function explicitly as `tf.nn.tanh(x)`, one receives the following misleading error message:

TypeError: in user code: TypeError: tf__call() got an unexpected keyword argument 'name'

Minimal example to reproduce:


import tensorflow as tf
import tensorflow_model_optimization as tfmot

i = tf.keras.layers.Input(shape=(24, 24, 3))
x = tf.keras.layers.Conv2D(10, kernel_size=1)(i)
x = tf.nn.tanh(x)
model = tf.keras.Model(inputs=i, outputs=x)

quant_aware_model = tfmot.quantization.keras.quantize_model(model)

