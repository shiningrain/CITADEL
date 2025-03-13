import tensorflow as tf

foo = tf.Variable(3.0)
ema = tf.train.ExponentialMovingAverage(0.1)
decayed_foo = ema.apply([foo])

