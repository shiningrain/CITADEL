import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(16, 1024))
dataset = dataset.apply(
    tf.data.experimental.snapshot('snapshot'))
dataset = dataset.shuffle(buffer_size=16)
dataset = dataset.batch(16)
dataset = dataset.repeat()
dataset = dataset.prefetch(1)
def run(dataset):
    iterator = iter(dataset)
    for _ in range(30):
        next(iterator)
for _ in range(10):
    run(dataset) 

If we run it with Tensorflow 2.4.0 (or Tensorflow 2.4.1), the output is:

...
2021-05-04 11:04:17.989897: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-05-04 11:04:17.990504: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2596985000 Hz
Segmentation fault (core dumped)

If either of `snapshot` or `repeat` or `prefetch` is removed, this would not occur.

**Describe the expected behavior**
The expected behavior is that there would not be a segmentation fault
**[Contributing](https://www.tensorflow.org/community/contribute)** - Do you
want to contribute a PR? (yes/no): - yes
Briefly describe your candidate solution
(if contributing):

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.
python
import tensorflow as tf
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(16, 1024))
dataset = dataset.apply(
    tf.data.experimental.snapshot('snapshot'))
dataset = dataset.shuffle(buffer_size=16)
dataset = dataset.batch(16)
dataset = dataset.repeat()
dataset = dataset.prefetch(1)
def run(dataset):
    iterator = iter(dataset)
    for _ in range(30):
        next(iterator)
for _ in range(10):
    run(dataset) 

