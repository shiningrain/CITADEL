tf.image.convert_image_dtype(image, dtype, saturate=False, name=None), for image and dtype parameters can be uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16. Then I set the image to complex64 and found it to work, so I don't know if the documentation is adequate.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
    image = tf.constant([[[254 + 2j]], [[83]], [[72]]], dtype=tf.complex64)
    dtype = tf.float64
    out = tf.image.convert_image_dtype(image, dtype)
    print(out)

