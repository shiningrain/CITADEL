import numpy as np
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import tensorflow as tf
func_cls(k=1070828000000, diagonal=np.ones((2,2,2,2)))

