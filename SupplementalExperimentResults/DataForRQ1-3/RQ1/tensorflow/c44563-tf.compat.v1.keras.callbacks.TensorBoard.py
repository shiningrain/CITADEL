import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np

tf.compat.v1.disable_eager_execution()

inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(1)(inp)

model = tf.keras.Model(inp, out)

model.predict(
    np.zeros((32, 1)),
    callbacks=[func_cls(log_dir="test", profile_batch=0)],
)

