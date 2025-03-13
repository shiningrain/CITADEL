import tensorflow as tf
import numpy as np

in_tensor = np.array([1, 2]).astype('uint32')  # would later cause abort() and core dump
shape = np.array([1e18, 2]).astype('int64') # make shape big enough to overflow int32
tf.broadcast_to(in_tensor, shape)


# import tensorflow as tf
# import numpy as np

# in_tensor = np.array([1, 2]).astype('float32')  # would later cause exception
# shape = np.array([1e18, 2]).astype('int64') # make shape big enough to overflow int32
# tf.broadcast_to(in_tensor, shape)

