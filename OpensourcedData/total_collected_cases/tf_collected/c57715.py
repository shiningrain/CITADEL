import numpy as np
import tensorflow as tf
tf.nn.conv2d_transpose(input=np.ones((1,1,1,1)), filters=np.ones((1,1,1,1)), output_shape=[2,-2], strides=[1])

