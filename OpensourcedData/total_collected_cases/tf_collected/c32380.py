import tensorflow as tf

inp = tf.keras.layers.Input(shape=(None, None, 3))
net = tf.keras.layers.BatchNormalization(virtual_batch_size=8)(inp)

model = tf.keras.Model(inputs=inp, outputs=net)

