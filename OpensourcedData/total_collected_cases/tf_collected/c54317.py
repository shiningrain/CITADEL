import tensorflow as tf

x = tf.complex(tf.random.uniform([4], dtype=tf.float64),tf.random.uniform([4], dtype=tf.float64))
print(tf.math.asin(x))
# Could not find device for node: {{node Asin}} = Asin[T=DT_COMPLEX128]

