import tensorflow as tf
A = tf.range(3)
tf.debugging.assert_shapes(((A, [3]),))  # works
# raises "ValueError: Attempt to convert a value (...) with an unsupported type (<class 'tensorflow.python.framework.sparse_tensor.SparseTensor'>) to a Tensor.
tf.debugging.assert_shapes(((tf.sparse.from_dense(A), [3]),))

