import tensorflow as tf

tf.debugging.enable_check_numerics()

x = tf.Variable(tf.random.uniform(shape=(10, 64, 64, 1), minval=0, maxval=1))

with tf.GradientTape() as tape:
    y = tf.image.ssim_multiscale(x, 1 - x, max_val=1, filter_size=11, filter_sigma=1.5, power_factors=(0.07105472, 0.45297383, 0.47597145), k1=0.01, k2=0.045)

tape.gradient(y, x)

