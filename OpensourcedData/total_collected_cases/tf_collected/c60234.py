I am trying to run a basic LSTM TFLite model with NNAPI delegate to explore the acceleration from Snapdragon 855's DSP Hexagon 690. I converted the simple LSTM model with full integer post-training quantization with intention to maximize hardware acceleration support, and ran this model on Pixel 4xl (snapdragon 855) with the latest pre-downloaded TFLite benchmark binary tool.

I am able to run other non-8bit model - env setup is correct. But 8bit model encountered error ` Unsupported input operand type for UNIDIRECTIONAL_SEQUENCE_LSTM op: TENSOR_QUANT8_ASYMM_SIGNED`.

I looked at NNAPI's operation support doc. It seems that operation `ANEURALNETWORKS_QUANTIZED_LSTM` is supported with the int8 inputs/outputs weights. But logcat suggests that TFLite NNAPI Model building process has converted the 8bit TFlite `UnidirectionalSequenceLSTM` to the non-quantized version `ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM`. Could this incorrect TFLite->NN operation conversion led to [this error](https://android.googlesource.com/platform/frameworks/ml/+/master/nn/common/operations/UnidirectionalSequenceLSTM.cpp#156)?



### Standalone code to reproduce the issue

shell
### To make the dummy TFLite model

def representative_dataset():
"""Just to make dummy input data for full integer quantization"""
    for _ in range(100):
        data = np.random.rand(8, 16)
        yield [data.astype(np.float32)]

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# make keras model
units = 512
batch_size = 8

model_in = Input(shape=(16,), batch_size=batch_size)
model = Model(model_in, Dense(units, activation="relu")(LSTM(736)(Embedding(4001, units,)(model_in))))

# convert to TFLite format
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter._experimental_default_to_single_batch_in_tensor_list_ops = True
tflite_model = converter.convert()

