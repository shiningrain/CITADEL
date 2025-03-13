import tensorflow as tf
import os
import numpy as np
try:
  pool_size_0 = 1e+38
  pool_size = [pool_size_0,]
  strides_0 = 2
  strides = [strides_0,]
  padding = "same"
  data_format = "channels_last"
  arg_class = tf.compat.v1.layers.MaxPooling1D(pool_size=pool_size,strides=strides,padding=padding,data_format=data_format,)
  arg_input_0_tensor = tf.random.uniform([1, 5, 4], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

