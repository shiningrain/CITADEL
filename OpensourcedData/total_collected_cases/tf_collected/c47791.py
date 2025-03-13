import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, Model 

embed_dim = 64; maxlen = 152; vocab_size = 8
K.clear_session()
X_tk = np.random.randint(1, vocab_size, (10, 152))
X_mask_tk = np.random.randint(1, vocab_size + 1, (10, 152)) #The +1 is for the mask token
l = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
# print(l(X_tk)) #this works
# print(l(X_mask_tk)) #this doesn't work

model = keras.Sequential([layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)])
# print(model(X_tk)) #this works
# print(model(X_mask_tk)) #this doesn't works

model2 = keras.Sequential([layers.Input(shape=(maxlen,)), layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)])
model2.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
# model2.fit(X_mask_tk, X_tk) #This doesn't work

strategy = tf.distribute.MirroredStrategy()
with strategy.scope(): #im using 2 gpus but I reckon this issue would occur even with 1 gpu

    model3 = keras.Sequential([layers.Input(shape=(maxlen,)), layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)])
    model3.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
model3.fit(X_mask_tk, X_tk) #This works

