import tensorflow as tf
inp = tf.keras.Input(shape=(10,))
out = tf.keras.layers.Lambda(
        lambda x_input: x_input,
        dynamic=True,
)(inp)
model = tf.keras.Model(inputs=inp, outputs=out)

