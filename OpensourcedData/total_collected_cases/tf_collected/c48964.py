import timeit
import math
import contextlib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import cupy


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def _gelu(x):
    constant = tf.constant(0.5 * (1 + math.tanh(math.sqrt(2 / math.pi))))
    x_cubed = tf.pow(x, 3, name="x_cubed")
    return tf.multiply(constant * x, tf.add(x, 0.044715 * x_cubed), name="gelu")


def gelu_graph(x):
    @tf.function
    def gelu(x):
        return _gelu(x)

    return gelu(x)


def gelu_eager(x):
    def gelu(x):
        return _gelu(x)

    return gelu(x)


# with options({'min_graph_nodes': 1}):
print(tf.config.optimizer.get_experimental_options())

x = tf.random.uniform((int(1e6), 128))

# with cupy.cuda.profile(): 
print("Graph execution:", timeit.timeit(lambda: gelu_graph(x), number=100), "s")

print("Eager execution:", timeit.timeit(lambda: gelu_eager(x), number=100), "s")

