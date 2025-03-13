import tensorflow as tf
import numpy as np

in_tensor = np.array([1, 2]).astype('uint32')  # would later cause abort() and core dump
shape = np.array([1e18, 2]).astype('int64') # make shape big enough to overflow int32
tf.broadcast_to(in_tensor, shape)

would cause the following:
> ...
2020-06-04 06:16:17.434615: F tensorflow/core/framework/tensor_shape.cc:345] Check failed: size >= 0 (-1486618624 vs. 0)
Aborted (core dumped)

When `input` is not `uint32` nor `uint64` and a `shape` big enough to overflow `int32`:
python
import tensorflow as tf
import numpy as np

in_tensor = np.array([1, 2]).astype('float32')  # would later cause exception
shape = np.array([1e18, 2]).astype('int64') # make shape big enough to overflow int32
tf.broadcast_to(in_tensor, shape)

