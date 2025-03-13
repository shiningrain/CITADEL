import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
import timeit

# make some random "images"
np_list = [np.random.randint(0, 256, (200, 200, 3)) for _ in range(1000)]

def convert_as_list():
    tensor = func_cls(np_list)

def convert_as_single_array():
    tensor = func_cls(np.asarray(np_list))

# just some operation to initialize
func_cls([1])

x1=timeit.Timer(convert_as_list).timeit(1)
x2=timeit.Timer(convert_as_single_array).timeit(1)

r_e_s=x1/x2

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
