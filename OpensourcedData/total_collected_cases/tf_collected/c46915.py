import tensorflow as tf
import numpy as np
tf.nn.atrous_conv2d(value=np.ones((1,1,1,5)), filters=np.ones((1,1,5,1)), rate=2147483647, padding='SAME')
