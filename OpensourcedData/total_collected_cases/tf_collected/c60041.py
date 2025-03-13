`zeta` doesn't support gradient computation. The Hurwitz zeta function is differentiable with respect to x.
The partial derivative with respect to x can be expressed as:

dζ(x, q) / dx = - ∑(n=0 to ∞) (n + q)^(-x) * ln(n + q)



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import numpy as np

x = tf.Variable(tf.constant([2], dtype=tf.float32))
y = tf.Variable(tf.constant([2], dtype=tf.float32))

def zeta(x, y):
    t = tf.math.zeta(x, y)
    return t

t = zeta(x, y)
print(t)
with tf.GradientTape() as tape:
    tape.watch(x)
    t = zeta(x, y)

gradient = tape.jacobian(t, x)
print(gradient)

