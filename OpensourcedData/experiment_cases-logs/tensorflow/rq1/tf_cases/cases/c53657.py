import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
logits = tf.random.uniform([16, 1, 10], dtype=tf.float16)
r1 = tf.nn.softmax(logits,axis=-1) # pass
logits_sp = tf.sparse.from_dense(logits)
r2 = func_cls(logits_sp) # InvalidArgumentError

