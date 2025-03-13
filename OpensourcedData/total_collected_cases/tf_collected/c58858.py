import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    BatchNormalization)
import numpy as np

inp = tf.keras.Input([1,192], batch_size = 1)
state_h = tf.keras.Input([192], batch_size = 1)
state_c = tf.keras.Input([192], batch_size = 1)
num_channels = 24

xT = BatchNormalization()(inp)

lstm_in = tf.keras.activations.tanh(xT)
x, new_states = tf.keras.layers.LSTMCell(xT.shape[2])(lstm_in[:,0,:], states = [state_h, state_c])
new_state_h = new_states[0]
new_state_c = new_states[1]
out = x + xT

my_model = tf.keras.Model(inputs=[inp, state_h, state_c],outputs = [ out, 
                                                                       new_state_h,
                                                                       new_state_c])

#save model and create tflite model
my_model.save('lstm_test', include_optimizer = False)
converter = tf.lite.TFLiteConverter.from_saved_model('lstm_test')
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        ]

tflite_model = converter.convert()

inpt = tf.random.normal([1,10,192])
init_state_h = tf.zeros([1,192])
init_state_c = tf.zeros([1,192])
init_state_h_tfl = tf.zeros([1,192])
init_state_c_tfl = tf.zeros([1,192])

#set up tflite interpreter
interpreter = tf.lite.Interpreter(model_content = tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input_details')
print(input_details)
print('output details')
print(output_details)

for k in range(10):
    outpt, st_h, st_c = my_model.predict([tf.expand_dims(inpt[:,k,:], axis=1), init_state_h, init_state_c])
    init_state_h = st_h
    init_state_c = st_c
    interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(inpt[:,k,:], axis = 1))
    interpreter.set_tensor(input_details[1]['index'], init_state_h_tfl)
    interpreter.set_tensor(input_details[2]['index'], init_state_c_tfl)
    interpreter.invoke()
    outpt_tfl = interpreter.get_tensor(output_details[0]['index'])
    st_h_tfl = interpreter.get_tensor(output_details[1]['index'])
    st_c_tfl = interpreter.get_tensor(output_details[2]['index'])
    init_state_h_tfl = st_h_tfl
    init_state_c_tfl = st_c_tfl
    np.testing.assert_almost_equal(outpt, outpt_tfl, decimal = 5)

