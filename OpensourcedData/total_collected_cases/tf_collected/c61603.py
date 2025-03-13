results = dict()
import tensorflow as tf
import os
import numpy as np
try:
  pool_size_0 = 1e+38
  pool_size_1 = 1048576
  pool_size = [pool_size_0,pool_size_1,]
  strides_0 = 2
  strides_1 = 2
  strides = [strides_0,strides_1,]
  padding = "same"
  data_format = None
  arg_class = tf.compat.v1.keras.layers.MaxPool2D(pool_size=pool_size,strides=strides,padding=padding,data_format=data_format,)
  arg_input_0_tensor = tf.random.uniform([3, 74, 74, 256], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))

print(results)

