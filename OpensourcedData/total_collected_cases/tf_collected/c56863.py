The jacobian matrix calculated in reverse mode are not equal to that in forward mode using tf.autodiff.ForwardAccumulator.



### Standalone code to reproduce the issue

shell
import numpy as np
import tensorflow as tf

input0 = tf.constant([[[[[[0.56601346]]],
   [[[0.35367298]]],
   [[[0.5609504 ]]]],
  [[[[0.43669665]]],
   [[[0.82903767]]],
   [[[0.57900476]]]]],
 [[[[[0.13786423]]],
   [[[0.17976725]]],
   [[[0.35320067]]]],
  [[[[0.34501767]]],
   [[[0.82709   ]]],
   [[[0.8386754 ]]]]],
 [[[[[0.8424399 ]]],
   [[[0.12519908]]],
   [[[0.41379738]]]],
  [[[[0.88551676]]],
   [[[0.26824057]]],
   [[[0.06636572]]]]]], shape=(3, 2, 3, 1, 1, 1), dtype=tf.float32)
input1 = tf.constant(
[[[[[[0.88215363],
     [0.56112957],
     [0.47048628]],
    [[0.23962319],
     [0.24418604],
     [0.68752027]],
    [[0.8854921 ],
     [0.8750253 ],
     [0.43920374]],
    [[0.6869767 ],
     [0.9971782 ],
     [0.21735716]]],
   [[[0.7472261 ],
     [0.7923174 ],
     [0.99001765]],
    [[0.23535097],
     [0.47414947],
     [0.53421795]],
    [[0.46127486],
     [0.31279147],
     [0.41679263]],
    [[0.40748405],
     [0.8575851 ],
     [0.62180364]]],
   [[[0.09478486],
     [0.8094814 ],
     [0.7278038 ]],
    [[0.36277568],
     [0.14143586],
     [0.6791742 ]],
    [[0.48797262],
     [0.34706163],
     [0.211653  ]],
    [[0.49032676],
     [0.37094796],
     [0.7821864 ]]]]]], shape=(1, 1, 3, 4, 3, 1), dtype=tf.float32)

softmax = tf.keras.layers.Softmax(axis=0)

input = [input0,input1]

with tf.GradientTape() as g:
    g.watch(input1)
    res_backward = softmax(*input)
grad = g.jacobian(res_backward,input1)

grad_fwd_arr = []

for i in range(tf.size(input1)):
    tangents = tf.reshape(tf.one_hot(i,tf.size(input1)),shape=input1.shape)
    with tf.autodiff.ForwardAccumulator(input1, tangents) as acc:
        res_forward = softmax(*input)
        jvp = acc.jvp(res_forward)
        grad_fwd_arr.append(jvp)

grad_fwd = tf.reshape(tf.convert_to_tensor(grad_fwd_arr),shape=grad.shape)

np.testing.assert_allclose(grad,grad_fwd)

