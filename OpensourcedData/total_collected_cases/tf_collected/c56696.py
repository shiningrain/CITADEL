tf.sparse.reshape can deal with tf.SparseTensor, however, it cannot deal with tf.keras.Input even if sparse=True is specified for the input.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
inputs = tf.keras.Input(shape=(3, 4), sparse=True, dtype=tf.int64)
output = tf.sparse.reshape(inputs, [-1, 4])

