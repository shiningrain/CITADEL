As per the raw_ops documentation https://www.tensorflow.org/api_docs/python/tf/raw_ops/DepthwiseConv2dNative
the DepthWiseConv2dNative op supports dilated convolutions. 

However, in practice, the operator seems to ignore the dilation values. Given any value for dilations, the output corresponds to dilation = [1, 1, 1, 1].



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

# CustomLayer using DepthwiseConv2dNative operator
class CustomDepthwiseConv2DLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomDepthwiseConv2DLayer, self).__init__()
    def build(self, input_shape):
        pass
    def call(self, input, filter, strides, padding, explicit_paddings, dilations):
        return tf.raw_ops.DepthwiseConv2dNative(
    input=input,
    filter=filter,
    strides=strides,
    padding=padding,
    explicit_paddings=explicit_paddings,
    dilations=dilations
)

# Model
inputs = tf.keras.Input(shape=(3, 3, 3))

filter_shape = (2, 2, 3, 1)
filter = tf.ones(filter_shape)
strides = [1, 1, 1, 1]
padding = "VALID"
explicit_paddings = []

# Enter any dilation value here. It has no effect!
dilations = [1, 1, 1, 1]
#dilations = [1, 2, 2, 1]
operator = CustomDepthwiseConv2DLayer()
outputs = operator(inputs, filter, strides, padding, explicit_paddings, dilations)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Prediction
x = np.arange(1, 28)
x = np.reshape(x, (1, 3, 3, 3))
x = tf.convert_to_tensor(x, dtype=tf.float32)

y = model.predict(x)

print(y.shape)
print(y)

