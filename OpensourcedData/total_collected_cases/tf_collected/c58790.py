When using `tf.math.cumsum(ragged_tensor, axis=axis)`, if `axis` is ragged and `ragged_tensor` contains `inf` or `nan`, the output will be `nan` for *all* the following flat values, even for those that are not supposed to be summed with those `nan`s. This doesn't happen for regular `Tensor`s. 

Moreover, the behavior persists when `exlusive=True` or `reverse=True` is passed to `cumsum()`. In the latter case, additional `nan`s occur before the problematic value, instead of after it.



### Standalone code to reproduce the issue

shell
import tensorflow as tf 
import numpy as np 

# RaggedTensor without NaNs or Infs
rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9, 2], [6, 5, 3]], dtype=tf.float32)
print('tf.math.cumsum(rt, axis=-1) = ', tf.math.cumsum(rt, axis=-1))

# RaggedTensor with Inf
rt2 = tf.ragged.constant([[3, 1, 4], [1, np.inf], [9, 2], [6, 5, 3]], dtype=tf.float32)
print('tf.math.cumsum(rt2, axis=-1) = ', tf.math.cumsum(rt2, axis=-1))
print('tf.math.cumsum(rt2.to_tensor(), axis=-1) = ', tf.math.cumsum(rt2.to_tensor(), axis=-1))
print('tf.math.cumsum(rt2, axis=-1, exclusive=True) = ', tf.math.cumsum(rt2, axis=-1, exclusive=True))
print('tf.math.cumsum(rt2.to_tensor(), axis=-1, exclusive=True) = ', tf.math.cumsum(rt2.to_tensor(), axis=-1, exclusive=True))
print('tf.math.cumsum(rt2, axis=-1, reverse=True) = ', tf.math.cumsum(rt2, axis=-1, reverse=True))
print('tf.math.cumsum(rt2.to_tensor(), axis=-1, reverse=True) = ', tf.math.cumsum(rt2.to_tensor(), axis=-1, reverse=True))

"""Expected output
tf.math.cumsum(rt, axis=-1) =  <tf.RaggedTensor [[3.0, 4.0, 8.0], [1.0, 6.0], [9.0, 11.0], [6.0, 11.0, 14.0]]>
tf.math.cumsum(rt2, axis=-1) =  <tf.RaggedTensor [[3.0, 4.0, 8.0], [1.0, inf],  [9.0, 11.0], [6.0, 11.0, 14.0]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1) =  tf.Tensor(
[[ 3.  4.  8.]
 [ 1. inf inf]
 [ 9. 11. 11.]
 [ 6. 11. 14.]], shape=(4, 3), dtype=float32)
tf.math.cumsum(rt2, axis=-1, exclusive=True) =  <tf.RaggedTensor [[0.0, 3.0, 4.0], [0.0, 1.0], [0.0, 9.0], [0.0, 6.0, 11.0]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1, exclusive=True) =  tf.Tensor(
[[ 0.  3.  4.]
 [ 0.  1. inf]
 [ 0.  9. 11.]
 [ 0.  6. 11.]], shape=(4, 3), dtype=float32)
tf.math.cumsum(rt2, axis=-1, reverse=True) =  <tf.RaggedTensor [[8.0, 5.0, 4.0], [inf, inf], [11.0, 2.0], [14.0, 8.0, 3.0]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1, reverse=True) =  tf.Tensor(
[[ 8.  5.  4.]
 [inf inf  0.]
 [11.  2.  0.]
 [14.  8.  3.]], shape=(4, 3), dtype=float32)
"""

"""Actual output
tf.math.cumsum(rt, axis=-1) =  <tf.RaggedTensor [[3.0, 4.0, 8.0], [1.0, 6.0], [9.0, 11.0], [6.0, 11.0, 14.0]]>
tf.math.cumsum(rt2, axis=-1) =  <tf.RaggedTensor [[3.0, 4.0, 8.0], [1.0, inf], [nan, nan], [nan, nan, nan]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1) =  tf.Tensor(
[[ 3.  4.  8.]
 [ 1. inf inf]
 [ 9. 11. 11.]
 [ 6. 11. 14.]], shape=(4, 3), dtype=float32)
tf.math.cumsum(rt2, axis=-1, exclusive=True) =  <tf.RaggedTensor [[0.0, 3.0, 4.0], [0.0, 1.0], [nan, nan], [nan, nan, nan]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1, exclusive=True) =  tf.Tensor(
[[ 0.  3.  4.]
 [ 0.  1. inf]
 [ 0.  9. 11.]
 [ 0.  6. 11.]], shape=(4, 3), dtype=float32)
tf.math.cumsum(rt2, axis=-1, reverse=True) =  <tf.RaggedTensor [[nan, nan, nan], [inf, inf], [11.0, 2.0], [14.0, 8.0, 3.0]]>
tf.math.cumsum(rt2.to_tensor(), axis=-1, reverse=True) =  tf.Tensor(
[[ 8.  5.  4.]
 [inf inf  0.]
 [11.  2.  0.]
 [14.  8.  3.]], shape=(4, 3), dtype=float32)
"""

