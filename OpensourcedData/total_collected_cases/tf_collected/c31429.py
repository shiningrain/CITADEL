import tensorflow as tf

image = tf.cast(tf.tile([[[0, 0, 0]]], [0, 0, 1]), tf.uint8)
# Or: = tf.cast(tf.reshape([], [0, 0, 3]))
try:
  tf.print(image)
  tf.print(tf.shape(image))
  tf.image.encode_png(image)
finally:
  print("We never get here!")

