import tensorflow as tf
data = tf.complex(tf.random.uniform([3, 4], dtype=tf.float64),tf.random.uniform([3, 4], dtype=tf.float64))
segment_ids = [0,0,1]
res = tf.math.segment_sum(data=data,segment_ids=segment_ids) # pass
res_sp = tf.sparse.segment_sum(data=data,indices=tf.constant([0, 1, 2]),segment_ids=segment_ids) # InvalidArgumentError

