import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
x = np.arange(9).reshape([1,3,3,1])
res = func_cls(x, size=[1023, -63], offsets=[1023, 63], centered=False, normalized=False) # Crash

