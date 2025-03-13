import tensorflow as tf
@tf.function
def use_tensor_array(x):
    y = tf.TensorArray(x.dtype, tf.shape(x)[0])
    for i in tf.range(tf.shape(x)[0]):
        y.write(i, x[i])
    y = y.stack()
    return y

use_tensor_array(tf.constant([1, 3, 4, 6]))

