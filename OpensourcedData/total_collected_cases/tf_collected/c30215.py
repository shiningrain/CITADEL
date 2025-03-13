import tensorflow as tf

c = tf.constant([5, 6, 7, 8, 9, 10], dtype=tf.int32)
d = tf.constant([5, 6, 7, 8, 9, 10], dtype=tf.int32)
x = tf.cast(c, dtype=tf.uint32)
y = tf.cast(c, dtype=tf.uint32)
with tf.Session() as sess:
    x_raw, y_raw = sess.run([x, y])
    print(x_raw.dtype, y_raw.dtype)
    print(x_raw)
    print(y_raw)
    
print(tf.__version__)

