import tensorflow as tf

try:
  tf.strings.unsorted_segment_join(inputs=['123'],segment_ids=[0],num_segments=-1)
except Exception:
  print('an exception should be thrown, but unsorted_segment_join crashes')

print('Not reached')

