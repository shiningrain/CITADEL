import os
import tensorflow as tf

# Force-disable GPU:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# The following line is how determinism will be enabled, but is not expected
# to currently have any effect on the output.
os.environ['TF_DETERMINISTIC_OPS'] = '1'

batch_size = 16
input_height = 64
input_width = 64
depth = 1
input_shape = (batch_size, input_height, input_width, depth)
tf.random.set_seed(456)
image = tf.random.uniform(
    input_shape, minval=-1.0, maxval=1.0, dtype=tf.float32)

# Image Gradients:
#
# Upsampling of a crop, leading to three or more output pixels being derived
# from an input pixel, will contribute to nondeterminism in the gradient
# associated with that input pixel location.
#
# Note that the number of boxes can be less than, equal to, or greater than
# the batch size. Three or more crops overlapping on the same input image pixel
# can independently contribute to nondeterminism in the image gradient
# associated with that input pixel location. This is independent of
# contributions caused by the upsampling of any given crop.
#
# Boxes Gradients:
#
# If the input and output dimensions are the same, then the boxes gradients
# will be deterministically zero, otherwise they will contain nondeterminism
# weather there is upsampling or downsampling and whether or not there are
# overlapping crops.

box_count = 4 * batch_size
boxes = tf.random.uniform(
  (box_count, 4), minval=0.0, maxval=1.01, dtype=tf.float32)

box_indices = tf.random.uniform(
  (box_count, ), minval=0, maxval=batch_size, dtype=tf.int32)

crop_size = [input_height*2, input_width*2]
output_shape = (box_count, *crop_size, depth)

injected_gradients = tf.random.uniform(
    output_shape, minval=-0.001, maxval=0.001, dtype=tf.float32)

def gradients():
  with tf.GradientTape() as tape:
    tape.watch([image, boxes])
    output = tf.image.crop_and_resize(
        image, boxes, box_indices, crop_size, method='bilinear')
    upstream = output * injected_gradients
  return tape.gradient(upstream, [image, boxes])

def sum(tensor):
  return tf.reduce_sum(tensor)

for device in ['gpu', 'cpu']:
  print("\n# Running on {}:\n".format(device))
  print("#         Image Gradients |  Boxes Gradients")
  print("#        -----------------+------------------")
  msg = "# Run {:d}: {:15.13f} | {:16.13f}"
  with tf.device("/{}:0".format(device)):
    for i in range(8):
      image_gradients, boxes_gradients = gradients()
      print(msg.format(i+1, sum(image_gradients), sum(boxes_gradients)))

print("")

# Example output (running on TensorFlow 2.3.0):

# Running on gpu:

#         Image Gradients |  Boxes Gradients
#        -----------------+------------------
# Run 1: -1.2592203617096 | -55.5643386840820
# Run 2: -1.2592202425003 | -55.5643081665039
# Run 3: -1.2592201232910 | -55.5643692016602
# Run 4: -1.2592203617096 | -55.5644264221191
# Run 5: -1.2592200040817 | -55.5643730163574
# Run 6: -1.2592201232910 | -55.5644035339355
# Run 7: -1.2592202425003 | -55.5643386840820
# Run 8: -1.2592201232910 | -55.5643272399902

# Running on cpu:

#         Image Gradients |  Boxes Gradients
#        -----------------+------------------
# Run 1: -1.2608621120453 | -55.5644989013672
# Run 2: -1.2792857885361 | -55.5644989013672
# Run 3: -1.2577195167542 | -55.5644989013672
# Run 4: -1.2553272247314 | -55.5644989013672
# Run 5: -1.2539256811142 | -55.5644989013672
# Run 6: -1.2622516155243 | -55.5644989013672
# Run 7: -1.2515500783920 | -55.5644989013672
# Run 8: -1.2537302970886 | -55.5644989013672

