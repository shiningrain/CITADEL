import tensorflow as tf
import numpy as np

a = np.array([[[1., 0., 2., 0.],
              [3., 0., 0., 4.]]])
b = (np.array([1., 2.])[:,np.newaxis]).T

a_t = tf.constant(a)
b_t = tf.constant(b)

a_s = tf.sparse.from_dense(a_t)

tf.sparse.sparse_dense_matmul(b_t,a_s)

