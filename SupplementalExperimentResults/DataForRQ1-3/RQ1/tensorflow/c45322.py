import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import os
import psutil
process = psutil.Process(os.getpid())

for i in range(int(1e7)):
    with func_cls("first"):
        with func_cls("second"):
            pass

r_e_s=process.memory_info().rss // 1000000

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)