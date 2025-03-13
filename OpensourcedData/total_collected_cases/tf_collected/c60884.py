import tensorflow as tf
import numpy as np

inp = tf.keras.Input([10,20], batch_size = 1, name = "input_0")
x = tf.keras.layers.LSTM(inp.shape[2],
                             return_sequences = True)(inp)
model_lstm = tf.keras.Model(inputs=inp, outputs=x)

rep_data = tf.data.Dataset.from_tensor_slices(np.float32(np.random.random_sample((10,1,10,20))))

def representative_dataset():
        for data in rep_data:
            yield {
            "input_0": data,
            }

converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS,
#comment line below to run at int 8
tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
converter.representative_dataset = representative_dataset

calibrated_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content = calibrated_model)
interpreter.allocate_tensors()

