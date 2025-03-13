import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def floordiv(x, y):
    # x // y
    return func_cls(x, y)

@tf.function
def floordiv_tffn(x, y):
    # x // y
    return func_cls(x, y)

@tf.function(experimental_compile=True)
def floordiv_compiled(x, y):
    # x // y
    return func_cls(x, y)

x, y = tf.constant([0., 0.1, 0.9]), 1.
x0=floordiv(x, y)
x1=floordiv_tffn(x, y)

r_e_s=x0==x1

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
