import tensorflow as tf
print(tf.__version__)
print(tf.version.GIT_VERSION, tf.version.VERSION)

model = tf.keras.Sequential()
model.add(tf.keras.Input((None,None,3)))
model.add(tf.keras.layers.SeparableConv1D(
                    dilation_rate=1,
                    filters = 3,
                    kernel_size=2,
                    padding='causal',
))

