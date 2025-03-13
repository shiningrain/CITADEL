# fails both with and without this environment variable set to disable GPU evaluation
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf


a_mat = np.array([
    [0., -1., -1.,  1.],
    [-1.,  0.,  1., -1.],
    [-1.,  1.,  0., -1.],
    [1., -1., -1.,  0.]], dtype=np.float32)


@tf.function(input_signature=[
    tf.TensorSpec(None, dtype=tf.float32)],
    jit_compile=False)
def eigh_uncompiled(arg):
    return tf.linalg.eigh(arg)


@tf.function(input_signature=[
    tf.TensorSpec(None, dtype=tf.float32)],
    jit_compile=True)
def eigh_compiled(arg):
    return tf.linalg.eigh(arg)


for eigh in [np.linalg.eigh, eigh_uncompiled, eigh_compiled]:
    val, vec = eigh(a_mat)
    # assert A.x = lambda x for eigen system
    print(tf.linalg.matmul(a_mat, vec) - val[tf.newaxis] * vec)
    if not np.allclose(tf.linalg.matmul(a_mat, vec), val[tf.newaxis] * vec, atol=1e-6):
        raise AssertionError(f'Test fails for function {eigh.__name__}')

