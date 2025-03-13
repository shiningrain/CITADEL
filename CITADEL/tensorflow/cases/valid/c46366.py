import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
layer = tf.keras.layers.LayerNormalization()
layer(tf.zeros([1, 0, 10]))

