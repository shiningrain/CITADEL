import tensorflow as tf
tf.distribute.experimental.CommunicationOptions(bytes_per_pack=-1) # This should fail!
communication = tf.distribute.experimental.CommunicationImplementation.NCCL
o = tf.distribute.experimental.CommunicationOptions(implementation=communication)
assert o.implementation == communication # Error: no attribute 'implementation'

