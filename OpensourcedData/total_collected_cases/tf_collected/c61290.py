loc(callsite(callsite(fused["Complex:", callsite("sequential/lambda/Complex@__inference__wrapped_model_30"("/opt/homebrew/lib/python3.11/site-packages/tensorflow/python/util/dispatch.py":1176:0) at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/python/util/traceback_utils.py":150:0 at callsite("/Users/drubinstein/test/complex_test/complex.py":4:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/layers/core/lambda_layer.py":212:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py":96:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/engine/base_layer.py":1150:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py":65:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/engine/sequential.py":420:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py":96:0 at "/opt/homebrew/lib/python3.11/site-packages/keras/src/engine/base_layer.py":1150:0)))))))))] at fused["PartitionedCall:", callsite("PartitionedCall@__inference_signature_wrapper_126"("/opt/homebrew/lib/python3.11/site-packages/tensorflow/python/saved_model/save.py":1313:0) at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/python/saved_model/save.py":1280:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py":1427:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/convert_phase.py":205:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py":1504:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py":1526:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py":1042:0 at callsite("/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py":1065:0 at "/Users/drubinstein/test/complex_test/complex.py":11:0))))))))]) at fused["PartitionedCall:", "PartitionedCall"])): error: 'tf.Complex' op is neither a custom op nor a flex op
error: failed while converting: 'main':
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select
TF Select ops: Complex
Details:
	tf.Complex(tensor<3xf32>, tensor<3xf32>) -> (tensor<3xcomplex<f32>>) : {device = ""}

Traceback (most recent call last):
  File "/Users/drubinstein/test/complex_test/complex.py", line 11, in <module>
    tflite_model = converter.convert()
                   ^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py", line 1065, in wrapper
    return self._convert_and_export_metrics(convert_func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py", line 1042, in _convert_and_export_metrics
    result = convert_func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py", line 1526, in convert
    saved_model_convert_result = self._convert_as_saved_model()
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py", line 1507, in _convert_as_saved_model
    return super(TFLiteKerasModelConverterV2, self).convert(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/lite.py", line 1296, in convert
    result = _convert_graphdef(
             ^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/convert_phase.py", line 212, in wrapper
    raise converter_error from None  # Re-throws the exception.
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/convert_phase.py", line 205, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/convert.py", line 918, in convert_graphdef
    data = convert(
           ^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/tensorflow/lite/python/convert.py", line 367, in convert
    raise converter_error
tensorflow.lite.python.convert_phase.ConverterError: /opt/homebrew/lib/python3.11/site-packages/tensorflow/python/util/dispatch.py:1176:0: error: 'tf.Complex' op is neither a custom op nor a flex op
<unknown>:0: note: loc(fused["PartitionedCall:", "PartitionedCall"]): called from
/opt/homebrew/lib/python3.11/site-packages/tensorflow/python/util/dispatch.py:1176:0: note: Error code: ERROR_NEEDS_FLEX_OPS
<unknown>:0: error: failed while converting: 'main':
Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select
TF Select ops: Complex
Details:
	tf.Complex(tensor<3xf32>, tensor<3xf32>) -> (tensor<3xcomplex<f32>>) : {device = ""}


**Standalone code to reproduce the issue** 

python3
import tensorflow as tf

model = tf.keras.models.Sequential(
    [tf.keras.layers.Lambda(lambda x: tf.dtypes.complex(x[0], x[1]))]
)

print(model([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

