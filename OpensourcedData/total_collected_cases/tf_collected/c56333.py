The documentation only states `tf.image.resize_with_pad` raises `ValueError`.
https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad#raises

The function actually raises `InvalidArgumentError` if the resized image becomes too small (zero width or zero height). The function should raise a `ValueError` corresponding to this situation, or at least document that the function raises `InvalidArgumentError`.

The same applies `tf.image.resize` with `preserve_aspect_ratio=True`.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
tf.image.resize_with_pad(tf.ones((1, 100, 1)), target_height=10, target_width=10)

