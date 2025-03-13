import cupy
import random

import tensorflow as tf

# gpu_to_use = 0      # Works
gpu_to_use = 1        # Errors

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_to_use], "GPU")

# Converting from TF to CuPy with dlpack works for both devices
tensor = tf.random.uniform((10,))

dltensor = tf.experimental.dlpack.to_dlpack(tensor)
array1 = cupy.fromDlpack(dltensor)

# Converting from CuPy to TF with dlpack only works for device 0
array1 = cupy.array([random.uniform(0.0, 1.0) for i in range(10)], dtype=cupy.float32)
dltensor = array1.toDlpack()
x = tf.experimental.dlpack.from_dlpack(dltensor)

# Using device 1 results in the following error

# Traceback (most recent call last):
#   File "examples/multi-gpu/tf-dlpack-repro.py", line 22, in <module>
#     x = tf.experimental.dlpack.from_dlpack(dltensor)
#   File "/home/karl/miniconda3/envs/nvtabular_dev_11.0/lib/python3.8/site-packages/tensorflow/python/dlpack/dlpack.py", line 66, in from_dlpack
#     return pywrap_tfe.TFE_FromDlpackCapsule(dlcapsule, context.context()._handle)
# tensorflow.python.framework.errors_impl.InvalidArgumentError: GPU:1 unknown device.

