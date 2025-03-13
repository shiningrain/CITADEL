import tensorflow as tf
v1 = tf.Variable(np.array([[1.,2.],[2.,3.]]))
v2 = tf.Variable(np.array([[1.,2.],[2.,3.]]))
v3=tf.keras.layers.Multiply()([v1,v2])

