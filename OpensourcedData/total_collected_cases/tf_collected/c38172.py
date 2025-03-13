import tensorflow as tf 
tf.signal.fftshift(tf.ones([1, 32, 32]), axes=[-2, -1])

