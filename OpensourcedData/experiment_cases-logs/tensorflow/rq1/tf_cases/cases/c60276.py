
import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
with tf.device("GPU:0"):
    validate_indices = False
    params = tf.saturate_cast(tf.random.uniform([13, 15, 7, 13, 14], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.half)
    indices = tf.saturate_cast(tf.random.uniform([11, 12, 6, 15, 11, 3], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int64)
    res = func_cls(
        validate_indices=validate_indices,
        params=params,
        indices=indices,
    )

