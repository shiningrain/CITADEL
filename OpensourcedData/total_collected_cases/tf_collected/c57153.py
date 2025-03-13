device_type is optional string on documentation. However, I find that when device_type is list/int/float or other type, code works. It just returns an empty list. So type restrictions should be removed from the documentation.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
results={}
try:
  arg_0 = -51
  results["res"] = tf.config.list_logical_devices(arg_0,)
except Exception as e:
  results["err"] = "Error:"+str(e)
print(results)
# results = {'res': []}

