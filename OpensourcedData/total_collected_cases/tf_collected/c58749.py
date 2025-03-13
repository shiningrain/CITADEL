The results of tf.image.convert_image_dtype running on CPU and GPU are very different.



### Standalone code to reproduce the issue

shell
CPU code:

    import tensorflow as tf
    with tf.device('/CPU'):
        arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        out = tf.image.convert_image_dtype(arg_0, dtype=tf.uint32, saturate=-1)
    print(out)

GPU code:

    import tensorflow as tf
    with tf.device('/GPU:0'):
        arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        out = tf.image.convert_image_dtype(arg_0, dtype=tf.uint32, saturate=-1)
    print(out)

