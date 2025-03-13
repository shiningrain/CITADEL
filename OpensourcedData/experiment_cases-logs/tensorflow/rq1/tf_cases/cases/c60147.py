import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np

init = np.random.rand(20)
update = np.random.rand(20)

resource = tf.Variable(init, dtype=tf.float32)
resource_var = resource.handle
indices = np.array([1, 3, 5], dtype=np.int32)
func_cls(resource=resource_var, indices=indices, updates=update)

