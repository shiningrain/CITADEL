
import os
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
print(tf.__version__)
for _ in range(20):
    try:
        with tf.device("GPU:0"):
            dense_input = tf.complex(tf.random.uniform([0, 1, 1], dtype=tf.float32),tf.random.uniform([0, 1, 1], dtype=tf.float32))
            indices = tf.saturate_cast(tf.random.uniform([1, 3], minval=0, maxval=64, dtype=tf.int64), dtype=tf.int64)
            res = func_cls(
                dense_input=dense_input,
                indices=indices,
            )
    except:
        pass

