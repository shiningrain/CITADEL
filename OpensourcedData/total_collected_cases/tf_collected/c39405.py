import tensorflow as tf


tf.tile(tf.constant([1,2,3], dtype=tf.uint64), (3, ))
tf.tile(tf.constant([1,2,3], dtype=tf.uint32), (3, ))
tf.repeat(tf.constant([1,2,3], dtype=tf.uint64), 3, axis=0)
tf.repeat(tf.constant([1,2,3], dtype=tf.uint32), 3, axis=0)

