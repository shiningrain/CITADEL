import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.utils.set_random_seed(1)
length = 5000
x = tf.concat([tf.ones([length, 1]), tf.random.normal([length, 2])], axis=1)
x = tf.tile(x[None, ...], [3, 1, 1])
xx = tf.matmul(x, x, transpose_a=True)
# xx = tf.einsum("ijk,ijm->ikm", x, x)  # Also doesn't work
print(f"{xx.numpy()}")


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
