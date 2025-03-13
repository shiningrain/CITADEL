import tensorflow as tf
x = tf.keras.Input([None, None, 16])
tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=32, dilation_rate=(1,2))(x)

