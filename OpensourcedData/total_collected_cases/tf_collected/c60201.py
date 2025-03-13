The following code can trigger a crash in `tf.raw_ops.DepthwiseConv2dNativeBackpropInput` due to check-fail in the latest version of TensorFlow.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
with tf.device("CPU"):
    strides = [1, 0, 1, 1]
    padding = "VALID"
    explicit_paddings = []
    data_format = "NHWC"
    dilations = [1, 0, 77, 1, 64]
    input_sizes = tf.saturate_cast(tf.random.uniform([3], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int32)
    filter = tf.random.uniform([16, 3, 3, 5], dtype=tf.bfloat16, minval=-1024, maxval=1024)
    out_backprop = tf.random.uniform([1, 0, 0, 1], dtype=tf.bfloat16, minval=-1024, maxval=1024)
    res = tf.raw_ops.DepthwiseConv2dNativeBackpropInput(
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        dilations=dilations,
        input_sizes=input_sizes,
        filter=filter,
        out_backprop=out_backprop,
    )

