import tensorflow as tf
num_rows = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)

num_columns = None
dtype = tf.qint16
y = tf.sparse.eye(num_rows, num_columns=num_columns, dtype=dtype, )
print(y)
x = tf.sparse.to_dense(y)

