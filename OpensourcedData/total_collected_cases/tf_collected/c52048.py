import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Net(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(6,3)
    
    def call(self, x):
        x = self.conv(x)
        return x

and profiling it on random input.
test_dummy_net.py


import tensorflow as tf
import dummy_net

inp = tf.random.uniform([1,300,300,3])
model = dummy_net.Net()
tf.profiler.experimental.start('./dummy_infer_logs')
y = model(inp)
tf.profiler.experimental.stop()
tf.debugging.set_log_device_placement(True)

