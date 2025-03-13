 import tensorflow as tf
 from tensorflow.python.ops import collective_ops
 FLAGS = tf.app.flags.FLAGS
 worker_replicas=2
 def shuffle(tensor):
    batch_size = tf.shape(tensor)[0]
    rank = FLAGS.task_index
    with tf.device('/cpu:0'):
        all_idx = tf.range(worker_replicas * batch_size)
        shuffle_idx = tf.random.shuffle(all_idx)
        if FLAGS.task_index == 0:
            index_broadcast = collective_ops.broadcast_send(shuffle_idx, shape=shuffle_idx.shape,
                                                            dtype=shuffle_idx.dtype,
                                                            group_size=worker_replicas, group_key=3, instance_key=100)
        else:
            index_broadcast = collective_ops.broadcast_recv(shape=shuffle_idx.shape, dtype=shuffle_idx.dtype,
                                                            group_size=worker_replicas, group_key=3, instance_key=100)
        my_idxs = tf.slice(index_broadcast, [rank * batch_size], [batch_size])
        all_tensor = collective_ops.all_gather(
            tensor, worker_replicas, worker_replicas, worker_replicas)

    return tf.gather(all_tensor, my_idxs), shuffle_idx



