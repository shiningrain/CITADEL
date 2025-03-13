import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
pool_size=[0, 2]
layer = func_cls(strides=1, pool_size=pool_size)
input_tensor = tf.random.uniform([3, 4, 10, 12], dtype=tf.float32)
res = layer(input_tensor) # crash
