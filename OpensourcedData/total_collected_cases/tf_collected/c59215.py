Training fails on following error when `tf.debugging.enable_check_numerics()` is enabled together with JIT compilation of model.

I was able to reproduce this with TF 2.9.2, 2.11.0 and current nightly build, on Tesla T4 and GeForce GTX 1080 Ti.


InvalidArgumentError: Graph execution error:

Detected unsupported operations when trying to compile graph __inference_run_step_1284[] on XLA_GPU_JIT: DebugNumericSummaryV2 (No registered 'DebugNumericSummaryV2' OpKernel for XLA_GPU_JIT devices compatible with node {{node DebugNumericSummaryV2}}){{node DebugNumericSummaryV2}}




### Standalone code to reproduce the issue

shell
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.debugging.enable_check_numerics()

# https://keras.io/examples/vision/mnist_convnet/

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 128
epochs = 15
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], jit_compile=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

