import tensorflow as tf 
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = tf.constant([1j], dtype=tf.complex64) 
print(func_cls(a))

