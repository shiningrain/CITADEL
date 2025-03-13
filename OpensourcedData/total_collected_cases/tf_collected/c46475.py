import gc
import os
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


input_tensor = tf.keras.layers.Input(shape=(512,64,1))

x = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding='same')(input_tensor)
# Commented out on purpose - see Note 1 below
# x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same')(x)
# Commented out on purpose - see Note 1 below
# x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=128, kernel_size=(4,4), strides=(2,1), padding='same')(x)
# Commented out on purpose - see Note 1 below
# x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=128, kernel_size=(4,4), strides=(2,1), padding='same')(x)
# Commented out on purpose - see Note 1 below
# x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Flatten()(x)

x = Dense(5, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_tensor, outputs=x)


train_x = np.random.random((2048, 512, 64, 1))
train_y = np.random.random((2048, 5))

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

process = psutil.Process(os.getpid())

for i in range(50):
    model.fit(train_x, train_y, epochs=1, batch_size=32, verbose=0)
    gc.collect()
    print(i, process.memory_info().rss // 1000000)

