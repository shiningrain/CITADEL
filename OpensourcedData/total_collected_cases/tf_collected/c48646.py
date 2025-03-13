import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
keras.layers.ReLU(max_value=1, threshold=-1, negative_slope=1, input_shape=(4,))])
x = tf.constant([[1.5, 0.5,-0.5, -1.5]])
print (model.predict(x,steps=1))

