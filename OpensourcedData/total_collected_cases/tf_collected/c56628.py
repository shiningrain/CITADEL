A bug happened during gradient computation of `tf.gather_nd` with `indices = [1,1]`. The code below is a minimized snippet. The forward pass of `tf.gather_nd` succeeds and the output `res` is correct. However, if I take the gradient,  it throws `InvalidArgumentError`.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np
params = tf.random.uniform([3, 2], dtype=tf.float32)
indices = [1,1]
res = tf.gather_nd(params, indices)
res = tf.reduce_sum(res)
print(res) # success

with tf.GradientTape(persistent=True,) as g:
  g.watch(params)
  res = tf.gather_nd(params, indices)
  res = tf.reduce_sum(res)
grad = g.gradient(res, params)
print(grad) # InvalidArgumentError

