import tensorflow as tf
for i in range(int(1e7)):
    with tf.name_scope("first"):
        with tf.name_scope("second"):
            pass

I'm actually not sure this is valid usage of name_scope. However, it seems this king of nested usage of name_scope is used in Tensorflow Probability ([here for example](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/distribution.py#L1542)). Another minimal example leading to memory leak is:

import tensorflow_probability as tfp
distr = tfp.distributions.Normal(loc=1.0, scale=2)
for i in range(int(1e7)):
    distr.sample()


I don't know why, but appending "/" to the name of the scope seems to fix the leak. Here are samples without memory leaks:

import tensorflow as tf
for i in range(int(1e7)):
    with tf.name_scope("first/"):
        with tf.name_scope("second"):
            pass

or

import tensorflow_probability as tfp
distr = tfp.distributions.Normal(loc=1.0, scale=2, name="foo/")
distr._name += "/"  # Force "/" because it's removed in the constructor
for i in range(int(1e7)):
    distr.sample(name="sample/")

