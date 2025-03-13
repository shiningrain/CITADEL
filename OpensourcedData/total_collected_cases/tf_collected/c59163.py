segfault happens with negative list elements.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_nn_ops
try:
  arg_0_tensor = tf.random.uniform([5, 20, 30, 3], dtype=tf.float64)
  arg_0 = tf.identity(arg_0_tensor)
  arg_1_0 = 2
  arg_1_1 = -5.267949192431123
  arg_1_2 = -52.58578643762691
  arg_1_3 = 1
  arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
  arg_2 = True
  arg_3 = True
  deterministic = True
  seed = 87654321
  seed2 = 341261001
  out = gen_nn_ops.fractional_avg_pool(arg_0,arg_1,arg_2,arg_3,deterministic=deterministic,seed=seed,seed2=seed2,)
except Exception as e:
  print("Error:"+str(e))



import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_nn_ops
try:
  arg_0_tensor = tf.random.uniform([1, 10, 10, 1], dtype=tf.float64)
  arg_0 = tf.identity(arg_0_tensor)
  arg_1_0 = True
  arg_1_1 = -0.35668935305391647
  arg_1_2 = -0.7209753581353426
  arg_1_3 = -87
  arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
  arg_2 = True
  arg_3 = True
  deterministic = True
  seed = 87654321
  seed2 = 341261001
  out = gen_nn_ops.fractional_avg_pool(arg_0,arg_1,arg_2,arg_3,deterministic=deterministic,seed=seed,seed2=seed2,)
except Exception as e:
  print("Error:"+str(e))

