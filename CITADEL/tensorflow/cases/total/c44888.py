import tensorflow as tf

tf.debugging.assert_shapes([(0, ())])  # passes as expected
tf.debugging.assert_shapes([(0, (1,))])  # passes unexpectedly

