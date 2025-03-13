**Custom quantized keras layer to build an example model**

import tensorflow as tf
from tensorflow import keras

class CustomConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, name="CustomConv2d"):
        super(CustomConv2D, self).__init__()
        self.w = self.add_weight(
            shape=(kernel_size, kernel_size, filters, filters), 
            initializer="random_normal", 
            dtype="float32", 
            name=self.name+"_weights", 
            trainable=True
        )
    
    def call(self, inputs):
        # Using the deprecated quantize_and_dequantize here since quantize_and_dequantize_v2 is listed as unsupported-ops by TF-TRT
        q_i = tf.quantization.quantize_and_dequantize(inputs, 0, 1, name=self.name+"_q_i", narrow_range=True)
        q_w = tf.quantization.quantize_and_dequantize(self.w, -1, 1, name=self.name+"q_w",narrow_range=True)
        return tf.nn.conv2d(q_i, q_w, 2, "SAME")
    

l = CustomConv2D(64, 3)
t = tf.random.normal((1, 224, 224, 64), dtype="float32")

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 64)))
for i in range(5):
    model.add(CustomConv2D(64, 3, name=f'custom_conv2d_{i}'))

model.save('./saved_model_qat/')


**Code used for converting saved quantized TF model using TF-TRT** 
python
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='saved_model_qat',
   precision_mode=trt.TrtPrecisionMode.INT8,
   use_calibration=False
)
trt_func = converter.convert()
converter.summary()

x_test = tf.ones((2, 224, 224, 64))


MAX_BATCH_SIZE=2
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size, :]
   yield [x]

converter.build(input_fn=input_fn)

