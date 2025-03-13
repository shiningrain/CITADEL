Check failure.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_nn_ops
try:
  arg_0_tensor = tf.random.uniform([1, 10, 14, 1], dtype=tf.float32)
  arg_0 = tf.identity(arg_0_tensor)
  ksize_0 = 1
  ksize_1 = 0
  ksize_2 = 3
  ksize_3 = 1
  ksize = [ksize_0,ksize_1,ksize_2,ksize_3,]
  strides_0 = 1
  strides_1 = 2
  strides_2 = 2
  strides_3 = 1
  strides = [strides_0,strides_1,strides_2,strides_3,]
  padding = "VALID"
  explicit_paddings = []
  data_format = "NHWC"
  out = gen_nn_ops.max_pool(arg_0,ksize=ksize,strides=strides,padding=padding,explicit_paddings=explicit_paddings,data_format=data_format,)
except Exception as e:
  print("Error:"+str(e))



import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_nn_ops
try:
  arg_0_tensor = tf.random.uniform([1, 6, 8, 1], dtype=tf.float32)
  arg_0 = tf.identity(arg_0_tensor)
  ksize_0 = 1
  ksize_1 = -52.0
  ksize_2 = 2
  ksize_3 = 1
  ksize = [ksize_0,ksize_1,ksize_2,ksize_3,]
  strides_0 = 1
  strides_1 = 1
  strides_2 = 1
  strides_3 = 1
  strides = [strides_0,strides_1,strides_2,strides_3,]
  padding = "VALID"
  explicit_paddings = []
  data_format = "NHWC"
  out = gen_nn_ops.max_pool(arg_0,ksize=ksize,strides=strides,padding=padding,explicit_paddings=explicit_paddings,data_format=data_format,)
except Exception as e:
  print("Error:"+str(e))

