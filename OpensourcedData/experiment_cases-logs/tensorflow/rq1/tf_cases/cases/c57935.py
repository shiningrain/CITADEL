import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
x = np.arange(10)
y = np.arange(10) 
print(np.floor_divide(0, 0)) # 0
with tf.device("gpu"):
  print(func_cls(0, 0)) # -1
with tf.device("cpu"):
  print(func_cls(0, 0)) # InvalidArgumentError
