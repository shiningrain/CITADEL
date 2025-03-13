import tensorflow as tf
tf.keras.utils.set_random_seed(1)
length = 5000
x = tf.concat([tf.ones([length, 1]), tf.random.normal([length, 2])], axis=1)
x = tf.tile(x[None, ...], [3, 1, 1])
xx = tf.matmul(x, x, transpose_a=True)
# xx = tf.einsum("ijk,ijm->ikm", x, x)  # Also doesn't work
print(f"{xx.numpy()}")

