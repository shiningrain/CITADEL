t1 = tf.Variable(-0.5, dtype=tf.float32)
tf.keras.layers.ReLU(max_value=1.0, threshold=-1.0)(t1)


results in 0, although I think, it should return -0.5.

**Code to reproduce the issue**

You can run the following script to see the problem (you'll need to install matplotlib).
`test_relu`  computes ReLU values in the manner described in the documentation.


from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

session = tf.Session()
xs = np.linspace(-3, 3, 100)

t1 = tf.Variable(0, dtype=tf.float32)
tf_ys = [session.run(tf.keras.layers.ReLU(max_value=1.0, threshold=-1.0)(t1), feed_dict={t1: x})
         for x in xs]


def test_relu(x, max_value, threshold, negative_slope):
    if x >= max_value:
        return max_value
    elif threshold <= x and x < max_value:
        return x
    else:
        return negative_slope * (x - threshold)


test_ys = [test_relu(x, 1, -1, 0) for x in xs]
plt.subplot(2, 1, 1)
plt.plot(xs, tf_ys)
plt.ylabel('tf.keras.layers.ReLU')
plt.subplot(2, 1, 2)
plt.plot(xs, test_ys)
plt.ylabel('test_relu')
plt.xlabel('x')
plt.show()

