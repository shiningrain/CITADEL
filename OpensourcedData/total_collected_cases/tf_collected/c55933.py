import tensorflow as tf

b = 1
n = 4

shape = [b,n,n]

# either # 
print(f"Einsum A*B: {tf.einsum('...ij, ...jk -> ...ik', tf.random.normal(shape), tf.random.normal(shape)).shape}")
print(f"Matmul A*B: {tf.matmul(tf.random.normal(shape), tf.random.normal(shape)).shape}") # <-- crash

# or # 
print(f"Matmul A*B: {tf.matmul(tf.random.normal(shape), tf.random.normal(shape)).shape}")
print(f"Einsum A*B: {tf.einsum('...ij, ...jk -> ...ik', tf.random.normal(shape), tf.random.normal(shape)).shape}")  # <-- crash

