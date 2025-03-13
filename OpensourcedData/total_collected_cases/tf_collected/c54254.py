import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' 

import tensorflow as tf

dataset = tf.data.Dataset.range(10)
    
@tf.function
def f():
    for x in dataset:
        tf.print(x)

with tf.device('/device:XLA_CPU:0'):
    f()

