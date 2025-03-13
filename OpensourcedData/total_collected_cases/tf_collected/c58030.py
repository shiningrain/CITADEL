tf.keras.metrics.CategoricalAccuracy crash with xla on cuda when dtype is set to copmlex



### Standalone code to reproduce the issue

shell
import tensorflow as tf
CategoricalAccuracy_class = tf.keras.metrics.CategoricalAccuracy(dtype=tf.complex64)

@tf.function(jit_compile=True)
def f(x0, x1, x2):
    return CategoricalAccuracy_class(x0, x1, x2)
x0 = [[0,0,1],[0,1,0]]
x1 = [[0.1, 0.1, 0.8], [0.05, 0, 0.95]]
x2 = [[0.5], [0.2]]
f(x0, x1, x2)

