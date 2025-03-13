import tensorflow as tf
input = tf.keras.Input(shape=(28, 28, 1), name='img',dtype=tf.bfloat16)
x = tf.keras.layers.UpSampling2D(3)(input)

