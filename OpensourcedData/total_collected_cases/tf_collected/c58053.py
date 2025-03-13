From code below, the tf.saturate_cast need to check dtype's min
https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/math_ops.py#L713-L743

However, from code below, min is not support for complex64 or complex128
https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/tensorflow_stub/dtypes.py#L188-L194

The documentation for tf.saturate_cast said it support any tensor or dtype desired. Therefore, either code or documentation need to be changed
https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/math_ops.py#L716-L729



### Standalone code to reproduce the issue

shell
Code like: 

import tensorflow as tf
arg_0_tensor = tf.complex(tf.random.uniform([2, 1], dtype=tf.float32),tf.random.uniform([2, 1], dtype=tf.float32))
arg_0_tensor = tf.saturate_cast(arg_0_tensor,dtype=tf.float32)

