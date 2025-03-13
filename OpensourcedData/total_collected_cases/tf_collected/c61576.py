import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import gen_image_ops
try:
  arg_0_tensor = tf.constant(-1610612736, shape=[0, 6, 6, 1], dtype=tf.bfloat16,)
  arg_0 = tf.identity(arg_0_tensor)
  arg_1_tensor = tf.constant(-45932682421089, shape=[2], dtype=tf.int32,)
  arg_1 = tf.identity(arg_1_tensor)
  align_corners = False
  out = gen_image_ops.resize_area(arg_0,arg_1,align_corners=align_corners,)
except Exception as e:
  print("Error:"+str(e))

