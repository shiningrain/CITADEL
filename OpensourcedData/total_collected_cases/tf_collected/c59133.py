Process get killed.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import array_ops
try:
  arg_0_0_tensor = tf.saturate_cast(tf.random.uniform([2147483654], minval=-128, maxval=128, dtype=tf.int64), dtype=tf.int8)
  arg_0_0 = tf.identity(arg_0_0_tensor)
  arg_0_1_tensor = tf.saturate_cast(tf.random.uniform([1024], minval=-128, maxval=128, dtype=tf.int64), dtype=tf.int8)
  arg_0_1 = tf.identity(arg_0_1_tensor)
  arg_0 = [arg_0_0,arg_0_1,]
  arg_1 = 0
  out = array_ops.concat(arg_0,arg_1,)
except Exception as e:
  print("Error:"+str(e))

