Forward AD threw error, but backward AD succeeded with same input. They should throw error both as expected. This behaviour can be reproduced in some similar APIs like `tf.keras.layers.AveragePooling2D` and `tf.keras.layers.AveragePooling3D`.



### Standalone code to reproduce the issue

shell
import tensorflow as tf

pool_size = [2,1,1]
strides = [2,2,2]
padding = "valid"
data_format = "channels_last"
input = tf.constant(0.895205,shape=[1,1,1,1,1], dtype=tf.float32)
layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, )

with tf.GradientTape(persistent=True, ) as g:
    g.watch(input)
    res_backward = layer(input)
grad_backward = g.jacobian(res_backward,res_backward)
print("res_backward:",res_backward)
print("grad_backward:",grad_backward)

tangents = tf.constant(1.,dtype=tf.float32,shape=[1,1,1,1,1])
with tf.autodiff.ForwardAccumulator(input, tangents) as acc:
    res_forward = layer(input)
    grad_jvp = acc.jvp(res_forward)
    print("res_forward:", res_forward)
    print("grad_forward", grad_jvp)

