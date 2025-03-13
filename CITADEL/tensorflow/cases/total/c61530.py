import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import array_ops
try:
  arg_0_tensor = tf.random.uniform([2, 3, 4, 5], dtype=tf.float32)
  arg_0 = tf.identity(arg_0_tensor)
  arg_1 = 0
  arg_2 = [()]
  range_given = True
  round_mode = "HALF_UP"
  axis = None
  out = array_ops.quantize_and_dequantize_v2(arg_0,arg_1,arg_2,range_given=range_given,round_mode=round_mode,axis=axis,)
except Exception as e:
  print("Error:"+str(e))

