import tensorflow as tf
import numpy as np
x = np.arange(10)
y = np.arange(10) 
print(np.floor_divide(0, 0)) # 0
with tf.device("gpu"):
  print(tf.experimental.numpy.floor_divide(0, 0)) # -1
with tf.device("cpu"):
  print(tf.experimental.numpy.floor_divide(0, 0)) # InvalidArgumentError



### Relevant log output

shell
0
tf.Tensor(-1, shape=(), dtype=int64)
InvalidArgumentError: {{function_node __wrapped__FloorDiv_device_/job:localhost/replica:0/task:0/device:CPU:0}} Integer division by zero [Op:FloorDiv]


For `remainder`:

### Code

import tensorflow as tf
import numpy as np
print(np.remainder(0, 0)) # 0
with tf.device("gpu"):
  print(tf.experimental.numpy.remainder(0, 0)) # 0
with tf.device("cpu"):
  print(tf.experimental.numpy.remainder(0, 0)) # InvalidArgumentError

