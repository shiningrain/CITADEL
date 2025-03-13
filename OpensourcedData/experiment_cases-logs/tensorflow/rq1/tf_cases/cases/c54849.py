import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
splits = [-16, 4, 2, 5, 5, 7]
result = func_cls(splits) # pass, but it should throw ValueError as splits starts with -16
r_e_s=result 


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
