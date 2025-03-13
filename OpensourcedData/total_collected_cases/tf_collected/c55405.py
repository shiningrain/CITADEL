import tensorflow as tf

labels = tf.ones((1, 50), dtype=tf.int64)
logits = tf.concat([tf.ones((1, 1000, 1)), -100 * tf.ones((1, 1000, 1))], axis=2)
label_length = tf.constant([50])
logit_length = tf.constant([1000])

dense_ctc_losses = tf.nn.ctc_loss(
    labels=labels,
    logits=logits,
    label_length=label_length,
    logit_length=logit_length,
    logits_time_major=False,
    blank_index=0,
)

sparse_ctc_losses = tf.nn.ctc_loss(
    labels=tf.sparse.from_dense(tf.cast(tf.convert_to_tensor(labels), dtype=tf.int32)),
    logits=logits,
    label_length=label_length,
    logit_length=tf.cast(logit_length, tf.int32),
    logits_time_major=False,
    blank_index=0,
)

ctc_losses = tf.cond(
    tf.math.reduce_all(tf.math.is_finite(sparse_ctc_losses)),
    lambda: sparse_ctc_losses,
    lambda: tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=0,
    ),
)

# This works fine and produces a proper value
print(f"Dense labels: {dense_ctc_losses}")
# Sparse version gives us an inf value
print(f"Sparse labels: {sparse_ctc_losses}")
# Uses sparse, falls back to dense if sparse gives Inf
print(f"Fallback CTC: {ctc_losses}")

