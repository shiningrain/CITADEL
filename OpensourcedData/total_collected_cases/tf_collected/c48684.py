import tensorflow as tf
cluster_spec = {"ps": ["ps0:2222"],"worker": ["worker0:2222", "worker1:2222"]}
with tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster_spec)):
    pass

