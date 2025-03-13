import tensorflow as tf
import numpy as np
x = np.arange(10)
y = np.arange(10) 
print(np.floor_divide(0, 0)) # 0
with tf.device("gpu"):
  print(tf.experimental.numpy.floor_divide(0, 0)) # -1
with tf.device("cpu"):
  print(tf.experimental.numpy.floor_divide(0, 0)) # InvalidArgumentError
