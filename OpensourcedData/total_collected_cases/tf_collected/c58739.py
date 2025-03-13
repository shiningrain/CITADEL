In the official document, the input types of tf.math.sqrt operator parameters are bfloat16, half, float32, float64.. It can be seen that the above does not contain complex64 or complex128 types, but in the following test code, the exception thrown In the information, you can see that the input can be of type complex64 or complex128. It is hoped that the documentation can be modified or the exception information can be modified.



### Standalone code to reproduce the issue

shell
    import tensorflow as tf

    input = "True"
    out = tf.math.sqrt(input)
    print(out)

