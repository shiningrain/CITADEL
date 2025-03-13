import tensorflow as tf

batch_size = 5

def key_func(element):
    return tf.size(element, out_type=tf.int64)

def reduce_func(key, dataset):
    return dataset.padded_batch(batch_size, padded_shapes=tf.TensorShape([None]))

dataset = tf.data.Dataset.from_tensor_slices(["a", "bb", "ccc"])
dataset = dataset.map(tf.strings.bytes_split)
dataset = dataset.group_by_window(key_func, reduce_func, window_size=batch_size)
next(iter(dataset))

