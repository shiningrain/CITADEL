import tensorflow as tf

def gen():
    with tf.device("CPU"):
        i = 0
        for i in tf.range(5000):
            i = tf.add(i, 1)
            print(i.device)
            yield i

d = tf.data.Dataset.from_generator(gen, output_types=(tf.int32))

for i in d:
    print(i)

Sample output of the above code:

...
/job:localhost/replica:0/task:0/device:GPU:0
tf.Tensor(4993, shape=(), dtype=int32)
/job:localhost/replica:0/task:0/device:CPU:0
tf.Tensor(4994, shape=(), dtype=int32)
/job:localhost/replica:0/task:0/device:GPU:0
tf.Tensor(4995, shape=(), dtype=int32)
/job:localhost/replica:0/task:0/device:CPU:0
tf.Tensor(4996, shape=(), dtype=int32)
/job:localhost/replica:0/task:0/device:GPU:0
tf.Tensor(4997, shape=(), dtype=int32
...

In some cases, the above code will also throw a RuntimeError: "Exiting device scope without proper scope nesting". I'll discuss why this happens later.

The following sample code **does not** reproduce the bug (which is expected, as discussed below):
(I moved the `tf.device` call inside the loop).

import tensorflow as tf

def gen():
    i = 0
    for i in tf.range(5000):
        with tf.device("CPU"):
            i = tf.add(i, 1)
            print(i.device)
            yield i

d = tf.data.Dataset.from_generator(gen, output_types=(tf.int32))

for i in d:
    print(i)

