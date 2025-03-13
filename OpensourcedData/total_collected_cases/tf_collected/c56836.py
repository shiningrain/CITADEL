Forward mode differentiation for the case below should be `0.0` but got `nan`, inconsistent with the gradient calculated in reverse mode.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
initial_learning_rate = 14.01
decay_steps = 100
decay_rate = 0.0
staircase = False
name = None

input = tf.constant(0.9369918869034664, dtype=tf.float64)

exp_decay = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,
                                                                  staircase=staircase, )

with tf.GradientTape(persistent=True) as g:
  g.watch(input)
  res_backward = exp_decay(input)
grad_backward = g.gradient(res_backward,input)
print(grad_backward)

with tf.autodiff.ForwardAccumulator(input, tf.constant(1,dtype=tf.float64)) as acc_0:
  res_forward = exp_decay(input)
grad_forward = acc_0.jvp(res_forward)
print(grad_forward)

