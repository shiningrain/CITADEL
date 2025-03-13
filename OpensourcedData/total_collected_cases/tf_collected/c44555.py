import tensorflow as tf
import numpy as np
import timeit

# make some random "images"
np_list = [np.random.randint(0, 256, (200, 200, 3)) for _ in range(1000)]

def convert_as_list():
    tensor = tf.convert_to_tensor(np_list)

def convert_as_single_array():
    tensor = tf.convert_to_tensor(np.asarray(np_list))

# just some operation to initialize
tf.convert_to_tensor([1])

print(f"convert_as_list: {timeit.Timer(convert_as_list).timeit(1)} s")
print(f"convert_as_single_array: {timeit.Timer(convert_as_single_array).timeit(1)} s")

