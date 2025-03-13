import tensorflow as tf

@tf.function
def my_func(x):
    with tf.control_dependencies([tf.debugging.assert_shapes([(x, (2,))])]):
        return x + 2

my_func([1, 2])
# TypeError: Can not convert a NoneType into a Tensor or Operation.

