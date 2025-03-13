
import tensorflow as tf
with tf.device("CPU"):
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NCDHW"
    dilations = [1, 1, 1, 1, 1]
    input = tf.random.uniform([1], dtype=tf.bfloat16, minval=-1024, maxval=1024)
    filter_sizes = tf.saturate_cast(tf.random.uniform([1], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int32)
    out_backprop = tf.random.uniform([2, 12, 3, 5, 10], dtype=tf.bfloat16, minval=-1024, maxval=1024)
    res = tf.raw_ops.Conv3DBackpropFilterV2(
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        input=input,
        filter_sizes=filter_sizes,
        out_backprop=out_backprop,
    )

