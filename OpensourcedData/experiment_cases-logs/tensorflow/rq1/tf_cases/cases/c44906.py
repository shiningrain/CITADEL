import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
inp = tf.keras.Input(shape=(10,))
out = func_cls(
        lambda x_input: x_input,
        dynamic=True,
)(inp)
model = tf.keras.Model(inputs=inp, outputs=out)

