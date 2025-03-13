padded mean:		1.5
unpadded mean:		3.0

padded variance:	3.25
unpadded variance:	2.0


**Describe the expected behavior**
The Boolean Mask should be used when calculating the Mean and Variance, to ignore any 0's as a result of padded data.

**Code to reproduce the issue**
This example compares the behavior of BatchNorm against a padded and un-padded 1D signal. The resulting Means and Variances come out very different.

import tensorflow as tf
import numpy as np

# Mask value of -10
padded_data = np.expand_dims(np.array([[1,2,3,4,5,-10,-10,-10,-10,-10] for _ in range(100)], dtype='float32'), 2)
data = np.expand_dims(np.array([[1,2,3,4,5] for _ in range(100)], dtype='float32'), 2)

def build_padding_model():
    input = tf.keras.layers.Input((10, 1))
    masked = tf.keras.layers.Masking(-10)(input)
    normed = tf.keras.layers.BatchNormalization(momentum=0.01)(masked)
    model = tf.keras.models.Model(input, normed)
    model.compile("adam", loss="mse")
    return model

def build_model():
    input = tf.keras.layers.Input((5, 1))
    normed = tf.keras.layers.BatchNormalization(momentum=0.01)(input)
    model = tf.keras.models.Model(inputs=input, outputs=normed)
    model.compile("adam", loss="mse")
    return model

if __name__ =='__main__':
    batch_size = 100
    signal_length = 20

    pad = build_padding_model()
    pad.fit(x=padded_data, y=padded_data, batch_size=10, epochs=5)

    nopad = build_model()
    nopad.fit(x=data, y=data, batch_size=10, epochs=5)

    weights1 = pad.layers[2].get_weights()
    weights2 = nopad.layers[1].get_weights()
    print('\npadded mean:\t\t' + str(weights1[2][0]) + '\nunpadded mean:\t\t' + str(weights2[2][0]) + '\n')
    print('padded variance:\t' + str(weights1[3][0]) + '\nunpadded variance:\t' + str(weights2[3][0]) + '\n')

