returns EagerTensor with shape [2,0]
**Describe the expected behavior**
tf.stack(list(np.ones([2,0,3]))).shape
returns EagerTensor with shape [2,0,3] which same as graph mode

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.
Eager mode:
 python
import tensorflow as tf
print(tf.stack(list(np.ones([2,0,3]))).shape)
tf.compat.v1.disable_eager_execution()
print(tf.stack(list(np.ones([2,0,3]))).shape)

