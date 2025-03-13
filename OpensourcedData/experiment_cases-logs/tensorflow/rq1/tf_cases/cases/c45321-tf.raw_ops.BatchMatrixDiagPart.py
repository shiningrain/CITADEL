import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

model_input = tf.keras.Input(shape=(3, 3), batch_size=1)

diagonal = func_cls(model_input, name='diag_part')

model = tf.keras.models.Model(inputs=model_input, outputs=diagonal)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable TensorFlow ops that are not directly supported by tf lite
# https://www.tensorflow.org/lite/guide/ops_select#convert_a_model
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
    ]

tflite_model = converter.convert()

