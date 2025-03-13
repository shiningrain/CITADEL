import numpy as np
%tensorflow_version 2.x
import tensorflow as tf
tf.__version__
boxes = np.array([[0.1,0.1,0.2,0.2], [0.3,0.3,0.3,0.4], [0.5,0.5,0.6,0.6], [0.7,0.7,0.8,0.8]], dtype= np.float32)
scores = np.array([0.9,0.8,0.7,0.6], dtype = np.float32)
(tf.image.non_max_suppression(boxes, scores, 8))


output:

'2.1.0-rc1'
<tf.Tensor: shape=(8,), dtype=int32, numpy=array([0, 1, 1, 1, 1, 1, 1, 1], dtype=int32)>


Now I ll explain brievely how the algorithm is coded in tensorflow. Given a list of candidate boxes containing in the beginning all the user boxes and a list of chosen boxes empty, if the box is chosen it will not immediately delete it from the candidate box. In fact, this box will be again processed as a candidate box in the next iteration. But because it is already in the chosen boxes it wont be chosen again . The reason for that is that the IOU of a box with himself is 1 which is always above the threshold. Unfortunately, the IOU of a line or a point with any box is 0. This is applied even when the IOU is calculated of the line with iteself. This will result in adding the line to the chosen boxes again and again. This behaviour is mentioned in this issue but wasn't clearly explained. https://github.com/tensorflow/tensorflow/issues/29628

**Describe the expected behavior**
The expected behaviour must be decided by the tensorflow programer. He can chose between putting it only once in the result : 
-<tf.Tensor: shape=(8,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)> 
or deleting the line box
 -<tf.Tensor: shape=(8,), dtype=int32, numpy=array([0, 2, 3], dtype=int32)>

- The second bug is really inexplainable. Why there is only the gpu specialisation of non_max_suppression_v2 ? Did the developer forgot about it? This was mentioned in several issues under the name : non max suppression work only on cpu. This is completely understandable because the default version of non_max_suppression is v3 which dosent have a gpu specialisation. 

**Code to reproduce the issue**
on colab u can just copy this code:


import numpy as np
%tensorflow_version 2.x
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
boxes = np.array([[0.1,0.1,0.2,0.2], [0.3,0.3,0.3,0.4], [0.5,0.5,0.6,0.6], [0.7,0.7,0.8,0.8]], dtype= np.float32)
scores = np.array([0.9,0.8,0.7,0.6], dtype = np.float32)
with tf.device('/GPU:0'):
  print(tf.image.non_max_suppression(boxes, scores, 8))

