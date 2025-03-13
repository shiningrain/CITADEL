import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

@tf.function
def my_func(x):
    with tf.control_dependencies([func_cls([(x, (2,))])]):
        return x + 2

my_func([1, 2])
# TypeError: Can not convert a NoneType into a Tensor or Operation.

