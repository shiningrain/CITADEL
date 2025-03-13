The documentation for `tf.raw_ops.RealDiv` states that the input argument `x` must be a Tensor of types `bfloat16, half, float32, float64, uint8, int8, uint16, int16, int32, uint32, uint64, int64, complex64, complex128.` Upon usage, however, this op throws an exception when run with an input of any integer dtype when run on a CPU. The op only works for `float` and `complex` dtypes.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

dtype = "int64"
x = np.array([[1,2,4],[2,3,5]], dtype=dtype)
y = np.array([[1,2,4],[2,3,5]], dtype=dtype)
x = tf.constant(x, dtype=dtype)
y = tf.constant(y, dtype=dtype)
tf.raw_ops.RealDiv(
    x=x, y=y, name=None
)

