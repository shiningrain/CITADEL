
import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
image = tf.constant([[[254 + 2j]], [[83]], [[72]]], dtype=tf.complex64)
dtype = tf.float64
out = func_cls(image, dtype)
r_e_s=out


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
