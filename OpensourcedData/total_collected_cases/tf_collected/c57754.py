tf.image.crop_and_resize crash (abort) when given num_boxes=0



### Standalone code to reproduce the issue

shell
import numpy as np
import tensorflow as tf
tf.image.crop_and_resize(crop_size=[1,1], box_indices=np.ones((0,1)), boxes=np.ones((0,4)), image=np.ones((2,2,2,2)))

