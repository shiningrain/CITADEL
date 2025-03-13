
import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np

dtype = "int64"
x = np.array([[1,2,4],[2,3,5]], dtype=dtype)
y = np.array([[1,2,4],[2,3,5]], dtype=dtype)
x = tf.constant(x, dtype=dtype)
y = tf.constant(y, dtype=dtype)
func_cls(
    x=x, y=y, name=None
)

