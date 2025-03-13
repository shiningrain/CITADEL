import tensorflow as tf
tf.boolean_mask([1, 2, 3], [True, False, True], axis=tf.constant(0, dtype=tf.int32))


Causes the exception:

none
TypeError: slice indices must be integers or None or have an __index__ method


For comparison, the equivalent operation with [`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather) works correctly:

py
import tensorflow as tf
with tf.Session() as sess:
    print(sess.run(tf.gather([1, 2, 3], [0, 1], axis=tf.constant(0, dtype=tf.int32))))
    # [1 2]



**Other info / logs**

Full traceback:

none
TypeError                                 Traceback (most recent call last)
<ipython-input-3-4beb5ed72842> in <module>
      1 import tensorflow as tf
----> 2 tf.boolean_mask([1, 2, 3], [True, False, True], axis=tf.constant(0, dtype=tf.int32))

~\AppData\Local\Continuum\anaconda3\envs\tf_test\lib\site-packages\tensorflow\python\ops\array_ops.py in boolean_mask(tensor, mask, name, axis)
   1369           " are None.  E.g. shape=[None] is ok, but shape=None is not.")
   1370     axis = 0 if axis is None else axis
-> 1371     shape_tensor[axis:axis + ndims_mask].assert_is_compatible_with(shape_mask)
   1372
   1373     leading_size = gen_math_ops.prod(shape(tensor)[axis:axis + ndims_mask], [0])

~\AppData\Local\Continuum\anaconda3\envs\tf_test\lib\site-packages\tensorflow\python\framework\tensor_shape.py in __getitem__(self, key)
    861     if self._dims is not None:
    862       if isinstance(key, slice):
--> 863         return TensorShape(self._dims[key])
    864       else:
    865         if self._v2_behavior:

TypeError: slice indices must be integers or None or have an __index__ method

