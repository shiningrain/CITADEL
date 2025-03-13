import tensorflow as tf
import numpy as np
import tempfile
from PIL import Image


def compress_PIL(img, quality):
    """
    Apply JPEG compression
    :param img: image with pixel intensities in range [0, 255]
    :return: img with intensities in range [0, 255]
    """

    if len(img.shape) == 3:
        # Remove singleton channel dimension
        img = np.squeeze(img, axis=2)

    # Apply JPEG compression
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        im = Image.fromarray(img)
        im.save(f.name, quality=quality)
        # Read back in
        im_recovered = Image.open(f.name)
        return np.array(im_recovered)


x = np.arange(64, dtype=np.uint8).reshape((8, 8, 1))

quality = 100

x_jpeg_tf = tf.image.adjust_jpeg_quality(x, quality)
print("TensorFlow")
print(x_jpeg_tf.numpy().squeeze())

x_jpeg_pil = compress_PIL(x, quality)
print("PIL")
print(x_jpeg_pil)

