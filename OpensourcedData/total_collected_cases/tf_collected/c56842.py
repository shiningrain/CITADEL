Jacobian matrix elements are not equal in forward mode and backward mode with the same input.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

shape = [2, 3]
seed = [7, 17]
means = 13.0
stddevs = tf.constant([[0.8059583, 0.09676647, 0.08382106],
                       [0.8149866, 0.44204712, 0.5636599]], dtype=tf.float32)
minvals = [-1.0, -2.0, -1000.0]
maxvals = [[10000.0], [1.0]]
name = None
with tf.GradientTape(persistent=True, ) as g:
  g.watch(stddevs)
  tf.random.set_seed(42)
  res_backward = tf.random.stateless_parameterized_truncated_normal(shape, seed, means=means, stddevs=stddevs,
                                                                    minvals=minvals, maxvals=maxvals, )
# shape=(2,3,2,3)
jacobian = g.jacobian(res_backward,stddevs)
print(jacobian[0][1])

tangents = tf.constant([[0.,1.,0.],
 [0.,0.,0.]],shape=(2,3),dtype=tf.float32)
with tf.autodiff.ForwardAccumulator(stddevs,tangents) as acc:
  res_forward = tf.random.stateless_parameterized_truncated_normal(shape, seed, means=means, stddevs=stddevs,
                                                                   minvals=minvals, maxvals=maxvals, )
print(acc.jvp(res_forward))

