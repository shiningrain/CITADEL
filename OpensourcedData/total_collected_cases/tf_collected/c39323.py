tf.Tensor(
[[[0.4200933  0.51168334 0.13771784]
  [0.4200933  0.51168334 0.13771784]
  [0.4200933  0.51168334 0.13771784]
  [0.31555724 0.80608404 0.38079023]
  [0.31555724 0.80608404 0.38079023]
  [0.22353566 0.7539935  0.28550136]]

 [[0.3753245  0.10351241 0.61035573]
  [0.51126313 0.4842764  0.5390732 ]
  [0.1071049  0.8601215  0.69413567]
  [0.1071049  0.8601215  0.69413567]
  [0.         0.         0.        ]
  [0.         0.         0.        ]]], shape=(2, 6, 3), dtype=float32)

**Standalone code to reproduce the issue**

import tensorflow as tf
@tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.int32),
    ])
def some_fz(x, dims):
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    dims = tf.cast(tf.math.round(dims), tf.int32)
    new_lengths = tf.reduce_sum(dims, axis=1)
    max_dim = tf.math.reduce_max(new_lengths)
    pad_sizes = max_dim - new_lengths
    new_batch = []
    x = tf.expand_dims(x, axis=-2)
    for i in tf.range(batch_size):
        tensor_list = []
        for j in tf.range(seq_len):
            tiled = tf.tile(x[i][j], [dims[i][j][0], 1])
            tensor_list.append(tiled)
        new_tensor = tf.concat(tensor_list, axis=0)  # breaks here
        new_tensor = tf.pad(new_tensor, [[0,pad_sizes[i][0]], [0,0]])
        new_batch.append(new_tensor)
    return tf.stack(new_batch)

if __name__ == '__main__':
    random = tf.random.uniform([2, 3, 3])
    vector = tf.constant([[[3], [2], [1]], [[1], [1], [2]]])
    out = some_fz(random, vector)

