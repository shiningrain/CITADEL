import tensorflow as tf
x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
tf.math.tanh(x)

