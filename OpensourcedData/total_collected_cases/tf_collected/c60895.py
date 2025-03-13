# tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:VE:0', device_type='VE'), PhysicalDevice(name='/physical_device:VE:1', device_type='VE'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


I traced down the error to be thrown here. It gets thrown in
https://github.com/tensorflow/tensorflow/blob/e32f5b90ec16e88b23be8a5189e52ea9a420e999/tensorflow/tsl/framework/device_id_utils.cc#L46

However, it is caused by wrong values stored in the gpu_options, which get initialized here:
https://github.com/tensorflow/tensorflow/blob/0db597d0d758aba578783b5bf46c889700a45085/tensorflow/python/eager/context.py#L1206

The list of gpu_devices and ALL pluggable_devices get combined, even if they are not of the same device_type. So the list of `compatible_devices` will be:

[
	PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
	PhysicalDevice(name='/physical_device:VE:0', device_type='VE'),
	PhysicalDevice(name='/physical_device:VE:1', device_type='VE')
]


This causes the `visible_device_list` to be `['0', '1', '2']`, which contains invalid GPU device indices. These then get passed to `ParseVisibleDeviceList`, which throws this error.

To fix this error, it suffices to change this line:
https://github.com/tensorflow/tensorflow/blob/0db597d0d758aba578783b5bf46c889700a45085/tensorflow/python/eager/context.py#L1216

and replace it to:
python
if dev not in gpu_devices and dev.device_type == "GPU":


This way, the list of `compatible_devices` will only populated with other GPUs, not with any other device types.

### Standalone code to reproduce the issue

shell
import tensorflow as tf
print(*tf.config.list_physical_devices(), sep='\n')
tf.device('/GPU:0')

