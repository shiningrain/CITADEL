import tensorflow as tf
import numpy as np
tf.nest.assert_same_structure(nest1=np.zeros((1)), nest2=tf.ones((1,1,1)), check_types=tf.ones((2)))
