lr_fn = PiecewiseConstantDecay()
opt = SGD(lr_fn)
opt = WrapOpt(opt)

We will hit the error of

InvalidArgumentError: Cannot assign a device for operation sequential_1/dense_1/Tensordot/ReadVariableOp: Could not satisfy explicit device specification '' because the node {{colocation_node sequential_1/dense_1/Tensordot/ReadVariableOp}} was colocated with a group of nodes that required incompatible device '/job:localhost/replica:0/task:0/device:GPU:0'. All available devices 

Note, this error seems to be hit only when PiecewiseConstantDecay schedule is used on GPUs.

**Describe the expected behavior**

We shouldn't see such error when using PiecewiseConstantDecay on GPUs.

**[Contributing](https://www.tensorflow.org/community/contribute)** - Do you
want to contribute a PR? (yes/no): - Briefly describe your candidate solution
(if contributing):

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.

Below is the colab link and please repro it with ***Runtime=GPU***.
https://colab.research.google.com/drive/1QPx4IqQNVpSR-ALfPYbjJjRUffyHo06G?usp=sharing

python
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models
print(tf.__version__)
class OptimizerWrapper(optimizers.Optimizer):
  def __init__(self, optimizer, name=None, **kwargs):
    super(OptimizerWrapper, self).__init__(name, **kwargs)
    self._optimizer = optimizer

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var):
    return self._optimizer._resource_apply_sparse(grad, var)

  def get_config(self):
    return self._optimizer.get_config()


model = tf.keras.Sequential()
model.add(layers.Dense(8))
x = tf.constant(12., shape=(5, 1, 2, 4))
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
#learning_rate_fn = optimizers.schedules.ExponentialDecay(
#    0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
#learning_rate_fn = optimizers.schedules.PolynomialDecay(
#    0.1, 10000, 0.01, power=0.5)
opt = optimizers.SGD(learning_rate=learning_rate_fn, momentum=1.0)
opt = OptimizerWrapper(opt)

@tf.function
def train_step(x):
  with tf.GradientTape(persistent=True) as tape:
    y = model(x)
    loss = tf.reduce_mean(y)

  grads = tape.gradient(loss, model.variables)
  opt.apply_gradients(zip(grads, model.variables))
  return loss

for i in range(3):
  loss = train_step(x)
  print("Loss:", loss)

