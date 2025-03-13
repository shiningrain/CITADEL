The documentation for `tf.image.ssim` claims to output

> ... a tensor containing an SSIM value for each pixel for each image in batch if return_index_map is True

However, the output image is smaller than the source image (see example below).

Upon comparing with a more well-known library implemented in PyTorch, I believe the reason for such discrepency is due to the Conv2D padding used.

In PyTorch's implementation, a "SAME" padding is used

https://github.com/Po-Hsun-Su/pytorch-ssim/blob/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L25

However, in the current Tensorflow implementation, "VALID" padding is used

https://github.com/tensorflow/tensorflow/blob/d5b57ca93e506df258271ea00fc29cf98383a374/tensorflow/python/ops/image_ops_impl.py#L4340

Please verify if this is the case.



### Standalone code to reproduce the issue

shell
import numpy as np
import tensorflow as tf # tf.__version__ == "2.11.0"
# B x T x n_mel
shape = (16, 2106, 80, 1)
image1 = np.arange(np.prod(shape))
image1 = (image1 / np.max(image1)) * 10 + 100
image1 = np.reshape(image1, shape)
image2 = np.linspace(0, 1, np.prod(shape))
image2 = np.exp(image2)
image2 = (image2 / np.max(image2)) * 10 + 100
image2 = np.reshape(image2, shape)

out_tf = tf.image.ssim(image1, image2, max_val=255, return_index_map=True)


