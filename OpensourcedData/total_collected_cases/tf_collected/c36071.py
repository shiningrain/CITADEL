import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from pprint import pprint
assert tf.__version__ == '2.1.0-rc1'

dummy_data = ["Foo", "bar", "foo foo", "foo bar", "foobar."]
predict_data = ["foo", "bar", "foobar", "foo foo", "OOV"]
inputs = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="text")
vectorize_layer = TextVectorization(output_mode="binary", max_tokens=5, split=None)
vectorize_layer.adapt(np.asarray(dummy_data))
print(f"Vocabulary:\t\t{vectorize_layer.get_vocabulary()}")
outputs = vectorize_layer(inputs)
model = tf.keras.Model(inputs, outputs)
print(f"Prediction data:\t{predict_data}")
predictions = model.predict(predict_data)
print(f"Predictions:")
pprint(predictions)

AttributeError                            Traceback (most recent call last)
<ipython-input-3-f1a03cb1e414> in <module>()
      9 inputs = tf.keras.layers.Input(shape=(1, ), dtype=tf.string, name="text")
     10 vectorize_layer = TextVectorization(output_mode="binary", max_tokens=5, split=None)
---> 11 vectorize_layer.adapt(np.asarray(dummy_data))
     12 print(f"Vocabulary:\t\t{vectorize_layer.get_vocabulary()}")
     13 outputs = vectorize_layer(inputs)

/tensorflow-2.1.0/python3.6/tensorflow_core/python/keras/layers/preprocessing/text_vectorization.py in _to_numpy(self, preprocessed_data)
    334     if isinstance(preprocessed_data, np.ndarray):
    335       return preprocessed_data
    336     return np.array(preprocessed_data.to_list())
AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'to_list'

