import tensorflow as tf
device_spec = tf.DeviceSpec(job="ps", device_type="CPU", device_index=0)
with tf.device(device_spec):
  pass

