import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(value=tf.ones((1,1,1,1)), seed=1, pooling_ratio=-1)
