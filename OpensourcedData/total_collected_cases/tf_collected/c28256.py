from PIL import Image
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

PATH = '/tmp/42313738-65c10f7c-807e-11e8-8f11-9db821e3c3cc.png'

im = Image.open(PATH)
ar = np.asarray(im)
pil_max = np.max(ar)
print(pil_max)

im = tf.gfile.FastGFile(PATH, 'rb').read()
ar = tf.image.decode_png(im, channels=1)
tf_max = tf.reduce_max(ar)
print(tf_max)

assert tf_max == pil_max

