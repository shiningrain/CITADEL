Documentation states that FusedBatchNormV3 should be able to work on bfloat16 input. However, if given bfloat16 input, the results are empty.

Minimal reproduction script that shows this OP works on float32 but does not work on bfloat16

Note that according to documentation, scale, offset, mean and variance have to be float32 and in fact TF throws an error if they are not, so the only variable that is bfloat16 is the input data x.

Results show that for bfloat16 output is an empty tensor of type float32.
This is similar to the task reported here https://github.com/tensorflow/tensorflow/issues/56680



### Standalone code to reproduce the issue

shell
import tensorflow as tf

for dtype in [tf.float32, tf.bfloat16]:
    x = tf.constant([[[[1],[5],[9],[13]]]], dtype=dtype)
    m_e = tf.constant([], dtype=tf.float32)
    v_e = tf.constant([], dtype=tf.float32)
    [y, m, v, _, _, _] = tf.raw_ops.FusedBatchNormV3(
        x=x, scale=[1.0], offset=[0.0], mean=m_e, variance=v_e
    )
    print("-----------------------------------------")
    print("dtype = ", dtype)
    print("in    = ", x)
    print("out y = ", y)
    print("out m = ", m)
    print("out v = ", v)
    print("")

