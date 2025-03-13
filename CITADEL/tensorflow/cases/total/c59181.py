
import tensorflow as tf
import numpy as np

dtype = "int64"
x = np.array([[1,2,4],[2,3,5]], dtype=dtype)
y = np.array([[1,2,4],[2,3,5]], dtype=dtype)
x = tf.constant(x, dtype=dtype)
y = tf.constant(y, dtype=dtype)
tf.raw_ops.RealDiv(
    x=x, y=y, name=None
)

