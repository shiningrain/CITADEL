import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(bytes_per_pack=-1) # This should fail!
communication = tf.distribute.experimental.CommunicationImplementation.NCCL
o = func_cls(implementation=communication)
assert o.implementation == communication # Error: no attribute 'implementation'

