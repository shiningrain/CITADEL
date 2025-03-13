import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
data = tf.complex(tf.random.uniform([3, 4], dtype=tf.float64),tf.random.uniform([3, 4], dtype=tf.float64))
segment_ids = [0,0,1]
res = func_cls(data=data,segment_ids=segment_ids) # pass
res_sp = tf.sparse.segment_sum(data=data,indices=tf.constant([0, 1, 2]),segment_ids=segment_ids) # InvalidArgumentError

