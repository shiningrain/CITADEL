$ python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"
2021-08-06 02:34:51.763077: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
unknown 2.4.0



**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.

Below is the minimum reproduce code snippet derived from 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/v1_compat_tests/session_ops_test.py#L249

python
"""Tests for tensorflow.ops.session_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import session_ops
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
import sys
import time

tf.debugging.set_log_device_placement(True)

def testFeedOneHandleDirectly():
  config = tf.compat.v1.ConfigProto()
  config.graph_options.optimizer_options.opt_level = -1
  config.allow_soft_placement = True
  config.graph_options.rewrite_options.constant_folding = (
      rewriter_config_pb2.RewriterConfig.OFF)
  config.graph_options.rewrite_options.pin_to_host_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  with tf.compat.v1.Session(config=config) as sess:

    a = constant_op.constant(10.0)
    b = constant_op.constant(5.0)
    c = math_ops.multiply(a, b)
    d = math_ops.multiply(c, c)

    h_c = sess.run(session_ops.get_session_handle(c))
    print(sess.run(d, feed_dict={c: h_c}))
    #print("result = %f " % d.eval())


testFeedOneHandleDirectly()


