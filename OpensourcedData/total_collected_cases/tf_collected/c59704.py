tf.image.convert_image_dtype behaves differently between CPU and GPU when inputs are negative numbers. CPU produces weird outputs but doesn't raise warnings or exceptions.



### Standalone code to reproduce the issue

shell
import tensorflow as tf

with tf.device("gpu"):

    x = tf.ones((10,10), dtype=tf.float32)*(-0.3)
    y = tf.image.convert_image_dtype(x, dtype=tf.uint8)
    print(y)

with tf.device("cpu"):

    x = tf.ones((10,10), dtype=tf.float32)*(-0.3)
    y = tf.image.convert_image_dtype(x, dtype=tf.uint8)
    print(y)

