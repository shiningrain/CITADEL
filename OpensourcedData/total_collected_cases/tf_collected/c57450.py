When I try to do back prop on Conv2dTranspose layer, strange `TypeError` is thrown. I think this bug is related to tensorflow because the error message contains tensorflow backprop ops.



### Standalone code to reproduce the issue

shell
import tensorflow as tf

filters = 2
kernel_size = [3, 3]
strides = [1, 1]
padding = "same"
output_padding = None
data_format = "channels_last"
dilation_rate = [2, 2]
activation = "linear"
use_bias = True
x = tf.random.uniform([1, 5, 6, 1], minval=0, maxval=1, dtype=tf.float32)
layer = tf.keras.layers.Convolution2DTranspose(filters, kernel_size, strides=strides, padding=padding, 
                                                                      output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, 
                                                                      activation=activation, use_bias=use_bias)


with tf.GradientTape() as g:
    g.watch(x)
    res = layer(x)
print(res.shape) # (1, 5, 6, 2)
grad = g.jacobian(res, x) # Error

