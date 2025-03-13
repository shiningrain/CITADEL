import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
from tensorflow.python.ops import gen_math_ops
try:
  try:
    with tf.device('/CPU'):
      arg_0_0_0 = -1.0
      arg_0_0_1 = 0.0
      arg_0_0_2 = 1.5
      arg_0_0 = [arg_0_0_0,arg_0_0_1,arg_0_0_2,]
      arg_0_1_0 = 2.0
      arg_0_1_1 = 5.0
      arg_0_1_2 = 15
      arg_0_1 = [arg_0_1_0,arg_0_1_1,arg_0_1_2,]
      arg_0 = [arg_0_0,arg_0_1,]
      arg_1_0 = -1.7976931348623157e+308
      arg_1_1 = -1.4013e-45
      arg_1 = [arg_1_0,arg_1_1,]
      arg_2 = 5
      dtype = tf.int32
      out = func_cls(arg_0,arg_1,arg_2,dtype=dtype,)
  except Exception as e:
    print("Error:"+str(e))
  try:
    with tf.device('/GPU:0'):
      arg_0_0 = [arg_0_0_0,arg_0_0_1,arg_0_0_2,]
      arg_0_1 = [arg_0_1_0,arg_0_1_1,arg_0_1_2,]
      arg_0 = [arg_0_0,arg_0_1,]
      arg_1 = [arg_1_0,arg_1_1,]
      dtype = tf.int32
      func_cls(arg_0,arg_1,arg_2,dtype=dtype,)
  except Exception as e:
    print("Error:"+str(e))
except Exception as e:
  print("Error:"+str(e))

