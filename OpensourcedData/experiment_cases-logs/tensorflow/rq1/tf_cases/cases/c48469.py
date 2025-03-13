import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
x = tf.keras.Input([None, None, 16])
func_cls(filters=1, kernel_size=32, dilation_rate=(1,2))(x)

