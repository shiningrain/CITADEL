EuclideanNorm: unsupported op: No registered 'EuclideanNorm' OpKernel for XLA_CPU_JIT devices compatible with node {{node EuclideanNorm}}


**Describe the expected behavior**

The op can run as usual.

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.

python3
import tensorflow as tf

x = tf.complex(tf.random.uniform(shape=(5, 5)), tf.random.uniform(shape=(5, 5)))

@tf.function(experimental_compile=True)
def reduce_euclidean_norm(x):
  return tf.math.reduce_euclidean_norm(x)

print(reduce_euclidean_norm(x))

