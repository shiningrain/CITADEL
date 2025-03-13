import tensorflow as tf


# eager execution since there's no @tf.function decorator
def graphcry():
    myscalar = tf.constant(83.2)  # just a random number

    with tf.name_scope('scalaragain scope'):  # doesn't work
        tf.summary.scalar('scalaragain', data=myscalar)

    with tf.name_scope('nospace_scope'):  # works
        tf.summary.scalar('nospace', data=myscalar)

graphcry()

