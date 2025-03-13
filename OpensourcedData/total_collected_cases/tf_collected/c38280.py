import tensorflow as tf
import tensorflow.keras as k

def construct_band_mat(band):
    # Dynamic batch size does not work
    batch_size = tf.shape(band)[0]

    # hard coded batch size works
    #batch_size = 3

    base = tf.zeros([batch_size, 5, 5])
    return tf.linalg.set_diag(base, band, k=(-1, 1))

input = k.Input(shape=[3, 5])
output = k.layers.Lambda(construct_band_mat)(input)

model = k.models.Model(inputs=[input], outputs=[output])

band_in = tf.ones([3, 3, 5], dtype=tf.float32)
print(model(band_in))

