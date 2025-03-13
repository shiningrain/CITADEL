import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.framework import dtypes
print(tf.__version__)
# tf.uint16, tf.uint32, tf.uint64
lhs = tf.constant([5, 0, 7, 11], dtype=tf.uint16)
rhs = tf.constant([5, 0, 7, 11], dtype=tf.uint16)
tf.assert_equal(lhs, rhs)

