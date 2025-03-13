import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(1, 1, colocate_gradients_with_ops=None, name='hessians')
