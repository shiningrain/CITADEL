import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import os
import numpy as np
try:
  arg_0_0 = 1e+38
  arg_0_1 = 16777216
  arg_0 = [arg_0_0,arg_0_1,]
  strides_0 = 2
  strides_1 = 2
  strides = [strides_0,strides_1,]
  padding = "same"
  arg_class = func_cls(arg_0,strides=strides,padding=padding,)
  arg_input_0_tensor = tf.random.uniform([3, 74, 74, 256], dtype=tf.float32)
  arg_input_0 = tf.identity(arg_input_0_tensor)
  arg_input = [arg_input_0,]
  out = arg_class(*arg_input)
except Exception as e:
  print("Error:"+str(e))


