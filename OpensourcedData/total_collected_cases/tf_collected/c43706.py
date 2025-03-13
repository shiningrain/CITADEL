import numpy as np
import tensorflow as tf

import time

# This would normaly come from some data stream, e.g. stream of TFRecords
some_tensor_list = [np.zeros(shape=(5,5), dtype=np.int32)]*100000
some_tensor_list_serialized = [tf.io.serialize_tensor(x) for x in some_tensor_list]

# Feed to tf.data
dataset = tf.data.Dataset.from_tensor_slices(some_tensor_list_serialized)

# Parsing whole batch back to tensors
def parse_batch(b):
    return tf.vectorized_map(lambda x: tf.io.parse_tensor(x, out_type = tf.int32), b)

# Parsing single record back to tensors
def parse_single(x):
    return tf.io.parse_tensor(x, out_type = tf.int32)

# Compare speed
def exaust_iterable(it):
    t = time.time()
    for _ in it:
        pass
    print(f'{time.time() - t}s')

# naive
dataset_naive = dataset.map(parse_single)

exaust_iterable(dataset_naive)

# vectorized over batch
dataset_vec = dataset.batch(32)
dataset_vec = dataset_vec.map(parse_batch)

exaust_iterable(dataset_vec)

