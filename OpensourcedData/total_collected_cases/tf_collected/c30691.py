import tensorflow as tf

xx = tf.constant([ 54043195528445964 , 72057594037927941 , 54043195528445957, 54043195528445954, 108086391056891910], dtype=tf.int64)
yy = tf.cast(xx, dtype=tf.uint64)


qqq=tf.constant([ 1,  2 , 3, 4, 5])
www=tf.constant([ 1,  2 , 3, 4, 5])

sess = tf.Session()
with sess.as_default():
    print(sess.run(xx))
    print(sess.run(yy))

the output is 

[ 54043195528445964  72057594037927941  54043195528445957
  54043195528445954 108086391056891910]
[0 0 0 0 0]

while it should have been

[ 54043195528445964  72057594037927941  54043195528445957
  54043195528445954 108086391056891910]
[ 54043195528445964  72057594037927941  54043195528445957
  54043195528445954 108086391056891910]


However, the following code

import tensorflow as tf

xx = tf.constant([ 54043195528445964 , 72057594037927941 , 54043195528445957, 54043195528445954, 108086391056891910], dtype=tf.int64)
yy = tf.cast(xx, dtype=tf.uint64)

sess = tf.Session()
with sess.as_default():
    print(sess.run(xx))
    print(sess.run(yy))

