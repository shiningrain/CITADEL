import tensorflow as tf
print(tf.__version__)
def model_float32():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, use_bias=False, input_shape=(10,)
        ,dtype=tf.float32))

    model.add(tf.keras.layers.GaussianNoise(0.0003))
    return model

testmodel_32 =model_float32()

without error.

2.0.0-beta1

 * However, the (`model_float64`) dose not work.
python
import tensorflow as tf
print(tf.__version__)
def model_float64():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, use_bias=False, input_shape=(10,)
        ,dtype=tf.float64))

    model.add(tf.keras.layers.GaussianNoise(0.0003))
    return model

testmodel_64 =model_float64()

