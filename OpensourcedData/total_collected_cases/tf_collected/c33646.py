import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Metric
import numpy as np

class CustomMetric(Metric):
  def __init__(self,
               name='score',
               dtype=tf.float32):
    super(CustomMetric, self).__init__(name=name)
    self.true_positives = self.add_weight(
        'true_positives',
        shape=[10],
        initializer='zeros',
        dtype=self.dtype)


  def update_state(self, y_true, y_pred, sample_weight=None):
    pass

  def result(self):
    return 0

  def get_config(self):
    """Returns the serializable config of the metric."""
    config = {}
    base_config = super(CustomMetric, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def reset_states(self):
    self.true_positives.assign(np.zeros(self.num_classes), np.float32)
    self.weights_intermediate.assign(
        np.zeros(self.num_classes), np.float32)
            
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=.001), metrics=[CustomMetric()])

model.save("model/", save_format='tf')

new_model = keras.models.load_model('model/',  tf.keras.models.load_model ={'score': CustomMetric})

