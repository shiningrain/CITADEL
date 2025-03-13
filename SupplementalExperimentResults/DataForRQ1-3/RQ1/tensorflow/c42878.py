import numpy as np
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

@tf.function(experimental_compile=True)
def test_tensor_array_scatter_gather():
    dtype = "float32"
    t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype))
    scatter_indices = tf.constant([2, 1, 0])
    gather_indices = tf.constant([1, 2])
    ta1 = func_cls(dtype=dtype, size=3, infer_shape=True)
    ta2 = ta1.scatter(scatter_indices, t)
    t1 = ta2.gather(gather_indices)
    return t1

test_tensor_array_scatter_gather()

