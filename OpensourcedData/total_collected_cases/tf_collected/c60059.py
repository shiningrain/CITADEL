QuantizeAndDequantizeV4 throws TypeError in gradient computation in forward mode. I encountered a similar issue (https://github.com/tensorflow/tensorflow/issues/59960) where quantize_and_dequantize_v2 is wrongly associated with _QuantizeAndDequantizeV4GradGrad in gradient computation. In this case, the V4 version API doesn't work either. It would be great if they could be fixed in the backend.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

tensor = tf.random.uniform(shape=[1, 1], dtype=tf.float32)

def quantizeAndDequantize(x):

    y = tf.raw_ops.QuantizeAndDequantizeV4(input=x, input_min=0., input_max=5.)
    return y

output = quantizeAndDequantize(tensor) # pass
with tf.autodiff.ForwardAccumulator(tensor, tf.constant([[0.], [1.]])) as acc:
    output = quantizeAndDequantize(tensor)

