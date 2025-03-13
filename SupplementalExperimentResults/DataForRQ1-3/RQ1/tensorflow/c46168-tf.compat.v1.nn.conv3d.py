import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
tf.config.experimental.enable_tensor_float_32_execution(False)
x_in = np.array([[
  [[2], [1], [2], [0], [1]],
  [[1], [3], [2], [2], [3]],
  [[1], [1], [3], [3], [0]],
  [[2], [2], [0], [1], [1]],
  [[0], [0], [3], [1], [2]], ]])
kernel_in = np.array([
 [ [[2, 0.1]], [[3, 0.2]] ],
 [ [[0, 0.3]],[[1, 0.4]] ], ])
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
out = func_cls(x, filters=kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', dilations=None)

