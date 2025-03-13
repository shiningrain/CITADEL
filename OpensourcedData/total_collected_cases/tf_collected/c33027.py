import tensorflow as tf
import tensorflow_probability as tfp

x = np.random.randn(1000, 1).astype(np.float32)

x = x * 0.001

bn = tf.keras.layers.BatchNormalization()

@tf.function
def _run_one(x):
    bn(x, training=True)

def run(x):
    for i in range(10000):
        _run_one(x)

tf.print(tf.math.reduce_std(bn(x, training=False), axis=0), tf.reduce_mean(bn(x, training=False), axis=0))
run(x)
tf.print(tf.math.reduce_std(bn(x, training=False), axis=0), tf.reduce_mean(bn(x, training=False), axis=0))

[0.000992934452] [-2.47328098e-05]
[0.0313995518] [-2.12341544e-09]


**Describe the expected behavior**


import tensorflow as tf
import tensorflow_probability as tfp

x = np.random.randn(1000, 1).astype(np.float32)

# x = x * 0.001
x = x * 2

bn = tf.keras.layers.BatchNormalization()

@tf.function
def _run_one(x):
    bn(x, training=True)

def run(x):
    for i in range(10000):
        _run_one(x)

tf.print(tf.math.reduce_std(bn(x, training=False), axis=0), tf.reduce_mean(bn(x, training=False), axis=0))
run(x)
tf.print(tf.math.reduce_std(bn(x, training=False), axis=0), tf.reduce_mean(bn(x, training=False), axis=0))


[2.03433657] [0.100093901]
[0.999882281] [1.81198118e-07]

