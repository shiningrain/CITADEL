Axis must be in the range [-rank(values), rank(values)) written on documentation [https://tensorflow.google.cn/versions/r2.4/api_docs/python/tf/concat#args]. When input is a list of tensor, axis will validate the range. However, when input is a single tensor, axis won't check the range.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
res = tf.concat([t1, t2], -100)
print(res)
'''
tensorflow.python.framework.errors_impl.InvalidArgumentError: ConcatOp : Expected concatenating dimensions in the range [-3, 3), but got -100 [Op:ConcatV2]
'''

Above code will check the validity of axis, but the following code wont.

import tensorflow as tf
results={}
try:
  arg_0 = tf.saturate_cast(tf.random.uniform([2, 4], minval=-256, maxval=257, dtype=tf.int64), dtype=tf.int64)
  axis = -200
  results["res"] = tf.concat(arg_0,axis=axis,)
except Exception as e:
  results["err"] = "Error:"+str(e)
print(results)
'''
{'res': <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
array([[ 140,  236, -151,   66],
       [ 154,   28,  -82,   23]])>}
'''

