import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
x1=func_cls(list(np.ones([2,0,3])), name='stack').shape
tf.compat.v1.disable_eager_execution()
x2=func_cls(list(np.ones([2,0,3]))).shape
r_e_s=x1==x2

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
