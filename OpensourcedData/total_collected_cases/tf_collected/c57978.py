tf.nn.max_pool raises 2 exceptions that are not mentioned in the documentation.
https://www.tensorflow.org/api_docs/python/tf/nn/max_pool

1. For 'input' tensor of rank 5, explicit padding is not supported.
This requirement is likely to be inferred by most users but maybe a mention in `Raises` section might be useful for some.


2. NCHW_VECT_C is not supported with explicit padding.
This is also the case for tf.nn.max_pool1d and tf.nn.max_pool2d. NCHW_VECT_C is mentioned as a valid option in a 'data_format' docstring of tf.nn.max_pool2d.

For max_pool1d, the documentation is probably ok as it is because users are unlikely to set `data_format` to NCHW_VECT_C.

For max_pool and max_pool2d, a suggestion is to revise the docstring of 'data_format', or add this requirement in a 'Raises' section, or remove NCHW_VECT_C from the documentation if it is not allowed in general.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
from tensorflow.python.ops import array_ops

x = array_ops.ones([2, 2, 2, 2, 2])
tf.nn.max_pool(x, ksize=2, strides=2, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])


### Relevant log output

shell
ValueError: Explicit padding is not supported with an input tensor of rank 5. Received: padding=[[0, 0], [1, 1], [1, 1], [0, 0]]


### Standalone code to reproduce the issue

shell
# Modified from max_pool's API doc example
import tensorflow as tf
from tensorflow.python.ops import array_ops

matrix = tf.constant([
    [0, 0, 1, 7],
    [0, 2, 0, 0],
    [5, 2, 0, 0],
    [0, 0, 9, 8],
])
reshaped = tf.reshape(matrix, (1, 4, 4, 1))
result = tf.nn.max_pool2d(reshaped, ksize=2, strides=2, padding=[[0, 0], [1, 1], [1, 1], [0, 0]], data_format="NCHW_VECT_C")

