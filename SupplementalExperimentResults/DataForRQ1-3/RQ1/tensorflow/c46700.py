import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
func_cls(input_tensor=1, keepdims=np.array([63600, 1], dtype=np.float16))
