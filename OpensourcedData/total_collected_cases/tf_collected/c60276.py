The following code can trigger a crash in `tf.raw_ops.Gather` due to check-fail in the latest version of TensorFlow.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
with tf.device("GPU:0"):
    validate_indices = False
    params = tf.saturate_cast(tf.random.uniform([13, 15, 7, 13, 14], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.half)
    indices = tf.saturate_cast(tf.random.uniform([11, 12, 6, 15, 11, 3], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int64)
    res = tf.raw_ops.Gather(
        validate_indices=validate_indices,
        params=params,
        indices=indices,
    )

