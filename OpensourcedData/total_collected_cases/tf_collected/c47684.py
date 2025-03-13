>>> x = tf.Variable([[1.2, 3.4]], shape=[None, 2])
>>> x.shape
TensorShape([None, 2])
>>> import copy
>>> copy.deepcopy(x).shape
TensorShape([5, 2])

