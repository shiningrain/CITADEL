import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
func_cls(size=1610637938, data_format='channels_first', interpolation='bilinear')(np.ones((5,1,1,1)))
