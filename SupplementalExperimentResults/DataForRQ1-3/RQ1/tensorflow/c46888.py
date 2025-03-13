
import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
func_cls(data=np.ones((1,10,1)), segment_ids=[1676240524292489355])
# tf.math.segment_min(data=np.ones((1,10,1)), segment_ids=[1676240524292489355])
# tf.math.segment_mean(data=np.ones((1,10,1)), segment_ids=[1676240524292489355])
# tf.math.segment_sum(data=np.ones((1,10,1)), segment_ids=[1676240524292489355])
# tf.math.segment_prod(data=np.ones((1,10,1)), segment_ids=[1676240524292489355])
