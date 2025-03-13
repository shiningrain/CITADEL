quantize_and_dequantize_v2 crashes when computing gradients in forward mode. According to the error message, `quantize_and_dequantize_v2` seems to link with the wrong version of the operator.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

tensor = tf.random.normal(shape=[1, 1], mean=127, stddev=50, dtype=tf.float32)

def func(tensor):
    quanti = tf.quantization.quantize_and_dequantize_v2(tensor, 
                                                        input_min = 0.0,
                                                        input_max = 255.0,
                                                        num_bits = 8,
                                                        range_given=False)
    
    return quanti

output = func(tensor) # pass
with tf.autodiff.ForwardAccumulator(tensor, tf.constant([[0.], [1.]])) as acc:
    output = func(tensor)

