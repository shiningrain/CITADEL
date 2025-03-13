tensorflow.python.framework.errors_impl.InvalidArgumentError:  Trying to add unsupported dtype 10
         [[node gradients/AddN_2 (defined at debug.py:44) ]] [Op:__inference___backward___backward_f_bad_373_1158_1489]

See full trace attached as [trace_without_pfor.txt](https://github.com/tensorflow/tensorflow/files/5471053/trace_without_pfor.txt).

The computation works fine when disabling  `tf.functions`.

First-order derivatives (gradient or jacobian) work fine too.

**Describe the expected behavior**

Hessian evaluates to `tf.Tensor([[2.]], shape=(1, 1), dtype=float32)`.

**Standalone code to reproduce the issue**

The following works if `use_function = False`, but fails for both `f_bad` and `f_bad_cond` when using `use_function = True`. Both `f_good` and `f_good_const` always work fine.


import tensorflow as tf

use_function = True
use_pfor = False

tf.config.run_functions_eagerly(not use_function)

@tf.function
def f_bad(x):
    if x < -1.:
        return tf.pow(x, 2)
    elif x <= 1.:
        return tf.pow(x, 2)
    else:
        return tf.pow(x, 2)

@tf.function
def f_bad_cond(x):
    return tf.cond(x < -1.,
                   lambda: tf.pow(x, 2),
                   lambda: tf.cond(x <= 1.,
                                   lambda: tf.pow(x, 2),
                                   lambda: tf.pow(x, 2)))

@tf.function
def f_good(x):
    if x < -1.:
        return tf.pow(x, 2)
    else:
        return tf.pow(x, 2)

@tf.function
def f_good_cond(x):
    return tf.cond(x < -1.,
                   lambda: tf.pow(x, 2),
                   lambda: tf.pow(x, 2))

f = f_bad

x = tf.Variable([0.])
with tf.GradientTape(persistent=not use_pfor) as t2:
    with tf.GradientTape() as t1:
        y = f(x)
    g_y = t1.gradient(y, x)
hess = t2.jacobian(g_y, x, experimental_use_pfor=use_pfor)
print(hess)

