>>> import tensorflow as tf
>>> scalar = tf.zeros(shape=())
>>> array = tf.zeros(shape=(1,))

>>> tf.random.uniform(shape=(),minval = scalar)
<tf.Tensor: id=25, shape=(), dtype=float32, numpy=0.021499991>

>>> tf.random.uniform(shape=(),minval = array)
<tf.Tensor: id=31, shape=(1,), dtype=float32, numpy=array([0.9388697], dtype=float32)>

