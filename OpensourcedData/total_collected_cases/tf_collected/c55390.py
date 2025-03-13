import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=0'

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

#with tf.compat.v1.device('gpu'):
ctr_y = tf.compat.v1.random.uniform([1], minval=8, maxval=9, dtype=tf.compat.v1.float32)
ctr_pred_ori = tf.compat.v1.tanh(ctr_y)


session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
session_config.graph_options.rewrite_options.disable_meta_optimizer=True
with tf.compat.v1.Session(config=session_config) as sess:
  while  True:
    ret = sess.run([ctr_pred_ori])[0]
    if ret > 1.0:
      print(ret);

