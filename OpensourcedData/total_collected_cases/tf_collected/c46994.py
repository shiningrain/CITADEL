import tensorflow as tf
import numpy as np
tf.nn.avg_pool1d(input=np.ones((1, 1, 1)), ksize=0, strides=1, padding='SAME')
tf.nn.avg_pool2d(input=np.ones((1, 1, 1, 1)), ksize=0, strides=1, padding='SAME')
tf.nn.avg_pool3d(input=np.ones((1, 1, 1, 1, 1)), ksize=0, strides=1, padding='SAME')
tf.nn.avg_pool(input=np.ones((1, 1, 1)), ksize=0, strides=1, padding='SAME')
