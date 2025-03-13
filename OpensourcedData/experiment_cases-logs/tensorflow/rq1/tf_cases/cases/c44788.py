import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

# The same function: decorated with tf.function (will be executed in graph mode) and 
# not decorated (will be executed in eager mode)

@tf.function
def graph_func(x):
    unique_input_ids, idx, counts = func_cls(x=x, axis=[0])
    tf.print('idx shape', tf.shape(idx), 'idx', idx)
    return tf.shape(idx)

def eager_func(x):
    unique_input_ids, idx, counts = func_cls(x=x, axis=[0])
    tf.print('idx shape', tf.shape(idx), 'idx', idx)
    return tf.shape(idx)

c = tf.constant([[0,0,1], 
                 [0,0,1], 
                 [0,0,2], 
                 [0,0,1]])
x1 = graph_func(c)
x2 = eager_func(c)

r_e_s=(x1==x2).numpy()

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
