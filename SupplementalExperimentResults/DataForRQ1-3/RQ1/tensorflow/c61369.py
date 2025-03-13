import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import pickle

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = func_cls(vocabulary=['a', 'b'])(model_input)
output = tf.keras.layers.Dense(10)(lookup)
full_model = tf.keras.Model(model_input, output)

# this part should work
model_bytes = pickle.dumps(full_model)
model_recovered = pickle.loads(model_bytes)


# this part should throw an error
full_model.save("/tmp/temp_model")
full_model_loaded = tf.keras.models.load_model("/tmp/temp_model")
model_bytes_2 = pickle.dumps(full_model_loaded)
model_recovered_2 = pickle.loads(model_bytes_2)

