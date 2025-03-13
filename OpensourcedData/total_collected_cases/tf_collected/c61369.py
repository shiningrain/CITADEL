full_model.layers[1].get_config()

Out[9]: {'name': 'string_lookup',
 'trainable': True,
 'dtype': 'int64',
 'invert': False,
 'max_tokens': None,
 'num_oov_indices': 1,
 'oov_token': '[UNK]',
 'mask_token': None,
 'output_mode': 'int',
 'sparse': False,
 'pad_to_max_tokens': False,
 'vocabulary': ListWrapper(['a', 'b']),
 'idf_weights': None,
 'encoding': 'utf-8'}


After saving and loading to the disk, we get:

full_model_loaded.layers[1].get_config()

Out[10]: {'name': 'string_lookup',
 'trainable': True,
 'dtype': 'int64',
 'invert': False,
 'max_tokens': None,
 'num_oov_indices': 1,
 'oov_token': '[UNK]',
 'mask_token': None,
 'output_mode': 'int',
 'sparse': False,
 'pad_to_max_tokens': False,
 'vocabulary': ListWrapper([]),
 'idf_weights': None,
 'encoding': 'utf-8'}
 

#-----------------------


We were able to circumvent the issue by creating a new class as follows:

@tf.keras.utils.register_keras_serializable()
class MyStringLookup(tf.keras.layers.StringLookup):
    def get_config(self):
        base_config = super().get_config()
        custom = {"vocabulary": self.get_vocabulary()}
        return {**base_config, **custom}


However, it would be nice if we didn't have to create this wrapper.


### Standalone code to reproduce the issue

shell
import tensorflow as tf
import pickle

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = tf.keras.layers.StringLookup(vocabulary=['a', 'b'])(model_input)
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

