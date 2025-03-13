The results of `LRNGrad` operators are inconsistent between CPU and GPU.

I've used two calls `tf.raw_ops.LRNGrad` and `nn.lrn_grad`,
and also changed the order of calls for different devices, 
still inconsistent results.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/lrn_op.cc

input_grads = random_ops.random_uniform(
        shape=[1, 1, 1, 3],
        minval=-10000,
        maxval=10000,
        dtype=tf.float32,
        seed=2022)
input_img   = random_ops.random_uniform(
        shape=[1, 1, 1, 3],
        minval=-10000,
        maxval=10000,
        dtype=tf.float32,
        seed=2022)
output_img  = random_ops.random_uniform(
        shape=[1, 1, 1, 3],
        minval=-10000,
        maxval=10000,
        dtype=tf.float32,
        seed=2022)


with tf.device('/GPU:0'):
    out = tf.raw_ops.LRNGrad(input_grads=input_grads, input_image=input_img, output_image=output_img)
    #out = nn.lrn_grad(input_grads=input_grads, input_image=input_img, output_image=output_img)
    print(out)

with tf.device('/CPU:0'):
    out = tf.raw_ops.LRNGrad(input_grads=input_grads, input_image=input_img, output_image=output_img)
    #out = nn.lrn_grad(input_grads=input_grads, input_image=input_img, output_image=output_img)
    print(out)

