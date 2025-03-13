import numpy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM

# uncommenting this causes performance issues
# tf.compat.v1.disable_eager_execution()

i = Input(shape=(1024, 32))
o = LSTM(units=32)(i)
m = Model(i, o)

m.compile('sgd', 'mse')
m.fit(numpy.zeros((512, 1024, 32)), numpy.zeros((512, 32)))

