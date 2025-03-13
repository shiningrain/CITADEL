import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
num_rows = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)

num_columns = None
dtype = tf.qint16
y = tf.sparse.eye(num_rows, num_columns=num_columns, dtype=dtype, )
print(y)
x = func_cls(sp_input=y)

