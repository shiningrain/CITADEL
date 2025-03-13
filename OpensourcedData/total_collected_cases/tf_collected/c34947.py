[1 1 5]
[[-2 1 10]]
[[8 2 20]]
[[-18 6 30]]
[[24 24 24]]


**Code to reproduce the issue**

import tensorflow as tf

x = tf.Variable([[-1.0], [0.0], [1.0]])

@tf.function
def func():
    with tf.GradientTape(persistent=True) as t:
        t.watch(x)
        coeffs = tf.eye(5)
        pv = tf.math.polyval(coeffs, x)
        y = tf.reduce_sum(pv, axis=1)
        dy_dx = t.gradient(y, x)
        d2y_dx2 = t.gradient(dy_dx, x)
        d3y_dx3 = t.gradient(d2y_dx2, x)
        d4y_dx4 = t.gradient(d3y_dx3, x)
    del t

    tf.print(y)
    tf.print(tf.transpose(dy_dx))
    tf.print(tf.transpose(d2y_dx2))
    tf.print(tf.transpose(d3y_dx3))
    tf.print(tf.transpose(d4y_dx4))

func()

