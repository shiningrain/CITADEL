import tensorflow as tf
tensor = [0,1,2,3]
mask = tf.random.uniform([4], dtype=tf.float64)
tf.boolean_mask(tensor, mask) 
# Outputs: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)>


**Describe the current behavior**
`tf.boolean_mask` has an argument `mask` which should be a `bool` tensor. However, it does not perform any validity checking and can accept a `float64` value. 


**Describe the expected behavior**
`tf.boolean_mask` should check the dtype of input tensor `mask`.

For example, `tf.math.reduce_any` would check the first argument and throw an `InvalidArgumentError` for non-boolean inputs.

import tensorflow as tf
input_tensor = tf.random.uniform([4], dtype=tf.float64)
tf.math.reduce_any(input_tensor) # InvalidArgumentError: cannot compute Any as input #0(zero-based) was expected to be a bool tensor but is a double tensor [Op:Any]

