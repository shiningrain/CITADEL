>>> import tensorflow as tf
>>> tf.keras.layers.ConvLSTM2D(1, 1).name
'conv_lst_m2d_0'


**Describe the expected behavior**
python
>>> import tensorflow as tf
>>> tf.keras.layers.ConvLSTM2D(1, 1).name
'conv_lstm_2d_0'


**Other info**
Problem caused here https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py#L2435
If a name is not provided to a `Layer` then it used the `generic_utils` function `to_snake_case`, which shows unexpected behavior here:
python
>>> from tensorflow.python.keras.utils import generic_utils
>>> generic_utils.to_snake_case('ConvLSTM2D')
'conv_lst_m2d'

