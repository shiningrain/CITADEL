import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
images = tf.random.uniform([1, 1, 3], dtype=tf.bfloat16)
func_cls(images=images)

