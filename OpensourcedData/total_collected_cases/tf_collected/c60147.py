tf.raw_ops.ResourceScatterUpdate crash with abortion



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

init = np.random.rand(20)
update = np.random.rand(20)

resource = tf.Variable(init, dtype=tf.float32)
resource_var = resource.handle
indices = np.array([1, 3, 5], dtype=np.int32)
tf.raw_ops.ResourceScatterUpdate(resource=resource_var, indices=indices, updates=update)

