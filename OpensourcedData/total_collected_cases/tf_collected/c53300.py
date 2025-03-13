import tensorflow as tf
x = tf.random.uniform(shape=[0,3])
y = tf.random.uniform(shape=[1,3])
print(tf.stack([x,y]).shape)

