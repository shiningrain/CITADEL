import os
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
layer = func_cls(beta_initializer='zeros', gamma_initializer='ones')
layer(tf.zeros([1, 0, 10]))

