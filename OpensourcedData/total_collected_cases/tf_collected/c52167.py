
import tensorflow as tf
import numpy as np
from PIL import Image

print(tf.__version__)

@tf.function
def tf_read_and_resize_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image=tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
    return image

# create random uint8 image
np.random.seed(42)
im = np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8)
im_filename = "/tmp/test_image.jpg"
im = Image.fromarray(im).save(im_filename)

# plain function
plain_tensor = tf_read_and_resize_image(im_filename)

# wrapped in tf.data map()
ds = tf.data.Dataset.from_tensor_slices([im_filename])
ds = ds.map(tf_read_and_resize_image)
ds_tensor = next(ds.as_numpy_iterator())

assert plain_tensor.dtype == ds_tensor.dtype == "float32"
assert plain_tensor.shape == ds_tensor.shape == (224, 224, 3)
np.testing.assert_array_equal(plain_tensor, ds_tensor) # fails with ~11% diff.

