import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

x = tf.constant([1, 2, 3])
func_cls(x, [110, 53, 104, 147, 157, 123, 5, 24, 188, 40, 5, 2])

