import tensorflow as tf
logits = tf.random.uniform([16, 1, 10], dtype=tf.float16)
r1 = tf.nn.softmax(logits,axis=-1) # pass
logits_sp = tf.sparse.from_dense(logits)
r2 = tf.sparse.softmax(logits_sp) # InvalidArgumentError

