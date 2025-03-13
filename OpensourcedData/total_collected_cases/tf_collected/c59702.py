tf.cast behaves differently in CPU and GPU when casting negative values



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

with tf.device('cpu'):
    data = np.array([[-1.0]]).astype(np.float64)
    x = tf.dtypes.cast(data, tf.uint8)
    print(x)
    
with tf.device('gpu'):
    data = np.array([[-1.0]]).astype(np.float64)
    x = tf.dtypes.cast(data, tf.uint8)
    print(x)

