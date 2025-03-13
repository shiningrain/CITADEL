import tensorflow as tf
# First batch:
# [?   e.]
# [1.  ? ]
# Second batch:
# [e   ? ]
# [e   e ]
shape = [2, 2, 2]  # 3-D SparseTensor
values = np.asarray([[[0., np.e], [1., 0.]], [[np.e, 0.], [np.e, np.e]]])
indices = np.vstack(np.where(values)).astype(np.int64).T

result = tf.sparse.softmax(tf.sparse.SparseTensor(indices, values, shape)) # ValueError
# ...returning a 3-D SparseTensor, equivalent to:
# [?   1.]     [1    ?]
# [1.  ? ] and [.5  .5]
# where ? means implicitly zero.

Outputs:

ValueError: Shape (2, 2, 2) must have rank 1


**Reason**
The `values` passed to `tf.sparse.SparseTensor` must be a rank-1 tensor instead of a rank-3 tensor.

**Fix**
The above code should be changed to:

import tensorflow as tf
shape = [2, 2, 2]  # 3-D SparseTensor
values = np.asarray([[[0., np.e], [1., 0.]], [[np.e, 0.], [np.e, np.e]]])
indices = np.vstack(np.where(values)).astype(np.int64).T
values = values[np.where(values)] # Flatten values
result = tf.sparse.softmax(tf.sparse.SparseTensor(indices, values, shape)) 
print(tf.sparse.to_dense(result))

