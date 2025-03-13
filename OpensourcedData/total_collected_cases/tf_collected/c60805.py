import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
def f(x):
    return tf.cond(tf.size(x) == 1,
                   # The reshape in this branch can only execute properly when the condition is true
                   lambda: tf.fill(tf.shape(x), tf.reshape(x, ())),
                   lambda: x)

# Works: this input is valid for both condition branches
tf.print(f(tf.constant([1])))
# [1]

# Fails: this input is only valid for the false branch, which is the active one
tf.print(f(tf.constant([1, 2])))
# tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error: (see below)


# If the shape in the function input signature is left completely undefined it works
# If the tf.function is defined with no input signature it works correctly as well

@tf.function(input_signature=[tf.TensorSpec(None, tf.int32)])
def f(x):
    return tf.cond(tf.size(x) == 1,
                   lambda: tf.fill(tf.shape(x), tf.reshape(x, ())),
                   lambda: x)

tf.print(f(tf.constant([1])))
# [1]

tf.print(f(tf.constant([1, 2])))
# [1 2]

