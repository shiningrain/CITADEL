t = np.array([[1 + 1j, 4 + 1j], [4 + 2j, 3 + 1j]])

tf.linalg.logdet outputs `-inf`, I think the expected output should be a normal value. For reference, PyTorch's torch.logdet outputs `tensor(2.6688-2.5536j, dtype=torch.complex128)`

### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np
t = np.array([[1 + 1j, 4 + 1j], [4 + 2j, 3 + 1j]])
tf_res = tf.linalg.logdet(tf.constant(t, dtype=tf.complex128))
torch_res = torch.logdet(torch.tensor(t))
print(tf_res, torch_res)

