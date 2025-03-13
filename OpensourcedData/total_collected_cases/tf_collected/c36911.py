import numpy as np
from tensorflow.python.ops.linalg.sparse import sparse as tfsp

a = np.ones((3, 3))
b = np.ones((5, 3, 3))
a_sp = tfsp.CSRSparseMatrix(a)
b_sp = tfsp.CSRSparseMatrix(b)

# This works fine
output = tfsp.matmul(a, b)
print(output.shape)

# These crash
output_sp = tfsp.matmul(a_sp, b)
output_sp = tfsp.matmul(a, b_sp)
output_sp = tfsp.matmul(a_sp, b_sp)

