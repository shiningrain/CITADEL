import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np


keras = tf.keras
layers = keras.layers


def infer(tflite_model, img):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details["index"], img)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    return output

def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    return tflite_model


# Image to test
np.random.seed(1)
input_shape = (12, 14, 18)
img = np.random.random((1,)+input_shape).astype(np.float32) * 7.0 - 3.5

# Create a model
i = layers.Input(shape=input_shape)
x = tf.quantization.fake_quant_with_min_max_args(i, min=0.0, max=3.984375)
x = func_cls(x, axis=None)
x = tf.quantization.fake_quant_with_min_max_args(x, min=-4.0, max=3.96875)
model = keras.models.Model(inputs=i, outputs=x)


tf_output = model.predict(img)
tflite_model = convert_to_tflite(model)
tflite_output = infer(tflite_model, img)

r_e_s=tf_output-tflite_output



import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)