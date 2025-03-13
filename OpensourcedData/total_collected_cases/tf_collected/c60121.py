tf.raw_ops.AssignAddVariableOp  crash with abortion



### Standalone code to reproduce the issue

shell
import tensorflow as tf
from tensorflow.python.eager import context

input1 = tf.raw_ops.VarHandleOp(dtype=tf.int32, shape=[2, 3], shared_name=context.anonymous_name())
input2 = tf.constant([],dtype=tf.float32)

tf.raw_ops.AssignAddVariableOp(resource=input1, value=input2)

