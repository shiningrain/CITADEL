When .max_pool_grad_with_argmax is given negative integer tensor, it crashes.



### Standalone code to reproduce the issue


import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_nn_ops
try:
  try:
    with tf.device('/CPU'):
      arg_0_tensor = tf.constant(-105687333925307, shape=[2, 3, 3, 1], dtype=tf.float32,)
      arg_0 = tf.identity(arg_0_tensor)
      arg_1_tensor = tf.random.uniform([2, 2, 2, 1], dtype=tf.float32)
      arg_1 = tf.identity(arg_1_tensor)
      arg_2_tensor = tf.random.uniform([2, 2, 2, 1], minval=-256, maxval=257, dtype=tf.int64)
      arg_2 = tf.identity(arg_2_tensor)
      ksize_0 = 1
      ksize_1 = 2
      ksize_2 = 2
      ksize_3 = 1
      ksize = [ksize_0,ksize_1,ksize_2,ksize_3,]
      strides_0 = 1
      strides_1 = 1
      strides_2 = 1
      strides_3 = 1
      strides = [strides_0,strides_1,strides_2,strides_3,]
      padding = "VALID"
      include_batch_in_index = False
      out = gen_nn_ops.max_pool_grad_with_argmax(arg_0,arg_1,arg_2,ksize=ksize,strides=strides,padding=padding,include_batch_in_index=include_batch_in_index,)
  except Exception as e:
    print("Error:"+str(e))
  try:
    with tf.device('/GPU:0'):
      arg_0 = tf.identity(arg_0_tensor)
      arg_0 = tf.cast(arg_0, tf.float32)
      arg_1 = tf.identity(arg_1_tensor)
      arg_1 = tf.cast(arg_1, tf.float32)
      arg_2 = tf.identity(arg_2_tensor)
      arg_2 = tf.cast(arg_2, tf.int64)
      ksize = [ksize_0,ksize_1,ksize_2,ksize_3,]
      strides = [strides_0,strides_1,strides_2,strides_3,]
      gen_nn_ops.max_pool_grad_with_argmax(arg_0,arg_1,arg_2,ksize=ksize,strides=strides,padding=padding,include_batch_in_index=include_batch_in_index,)
  except Exception as e:
    print("Error:"+str(e))
except Exception as e:
  print("Error:"+str(e))

