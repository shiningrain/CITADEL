import os
import numpy as np
import keras.layers as L
from keras.models import load_model
from keras.engine import Model, Input

kwargs = {'filters': 19, 'kernel_size': 0, 'padding': 'valid', 'strides': (2, 4), 'dilation_rate': 1, 'data_format': 'channels_first'}
input = (10 * np.random.random((1,32,32,16)))
layer = L.convolutional.Conv2D(**kwargs)
x = Input(batch_shape=input.shape)
y = layer(x)
bk_model = Model(x, y)
model_path = os.path.join('./', 'model.h5')
bk_model.save(model_path, bk_model)
model = load_model(model_path)
output = model.predict(input)
print('finish')

