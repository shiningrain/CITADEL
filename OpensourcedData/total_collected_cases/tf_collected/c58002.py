Running this code:
`tf.tile(tf.ones((3,)),(2,1))` will currently give an error for TensorFlow since the dimension length of the input and the multiples is different. Numpy and Pytorch handle this by modifying the dimension of the multiples or the input argument. I believe it would be nice if TensorFlow did this too.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
tf.tile(tf.ones((3,)),(2,1))

