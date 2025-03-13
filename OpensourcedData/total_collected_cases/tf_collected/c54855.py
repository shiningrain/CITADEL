import tensorflow as tf
images = tf.random.uniform([1, 1, 3], dtype=tf.bfloat16)
tf.raw_ops.RGBToHSV(images=images)

