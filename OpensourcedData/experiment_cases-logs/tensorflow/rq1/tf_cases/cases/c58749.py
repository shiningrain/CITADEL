


import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
with tf.device('/CPU'):
    arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    x1 = func_cls(arg_0, dtype=tf.uint32, saturate=-1).numpy()




import tensorflow as tf
with tf.device('/GPU:0'):
    arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    x2 = func_cls(arg_0, dtype=tf.uint32, saturate=-1).numpy()
r_e_s=x1-x2


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
