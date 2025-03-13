import numpy as np
import tensorflow as tf


def gen():
    for i in range(2 ** 13):
        array = np.random.random_sample(1024*1024*4).reshape(
            (1024, 1024, 4)).astype(np.float32)
        yield array * 1024 # Exacerbate the issue.

dataset = tf.data.Dataset.from_generator(
    gen, tf.float32, tf.TensorShape([1024, 1024, 4]))

dataset = dataset.batch(4)

norm = tf.keras.layers.experimental.preprocessing.Normalization()

norm.adapt(dataset)             # This ends up with RuntimeWarnings.

print(norm.mean)                  # Result is all 'inf'.
print(norm.variance)              # Result is 0.

