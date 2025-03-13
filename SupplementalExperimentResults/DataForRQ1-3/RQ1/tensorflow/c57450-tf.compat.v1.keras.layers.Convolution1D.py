import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

filters = 2
kernel_size=[3]
strides=[1]
padding = "same"
output_padding = None
data_format = "channels_last"
dilation_rate=[2]
activation = "linear"
use_bias = True
x = tf.random.uniform([1, 5, 1], minval=0, maxval=1, dtype=tf.float32)
layer = func_cls(filters, kernel_size, strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias)


with tf.GradientTape() as g:
    g.watch(x)
    res = layer(x)
print(res.shape) # (1, 5, 6, 2)
grad = g.jacobian(res, x) # Error

