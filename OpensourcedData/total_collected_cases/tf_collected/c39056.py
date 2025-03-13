%tensorflow_version 2.x
import tensorflow as tf

@tf.function
def example_write():
  writer = tf.io.TFRecordWriter("test.tfr")
  x = tf.constant([[0, 1], [2, 3]])
  x = tf.io.serialize_tensor(x)
  feature = {
      "data": tf.train.Features(
        bytes_list=tf.train.BytesList(value=[x]))
  }
  ex = tf.train.Example(features=tf.train.Features(
      feature=feature))
  writer.write(ex.SerializeToString())

example_write()

