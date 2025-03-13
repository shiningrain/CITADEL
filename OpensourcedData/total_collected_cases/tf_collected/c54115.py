import tensorflow as tf
image = [[[1, 2, 3], [4, 5, 6]],
         [[7, 8, 9], [10, 11, 12]],
         [[13, 14, 15], [16, 17, 18]]]
image = tf.constant(image)
tf.image.adjust_hue(image, -2)
tf.image.adjust_hue(image, 2)


