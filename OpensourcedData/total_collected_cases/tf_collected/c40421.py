import numpy as np
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

# The message that the cuDNN GRU implementation is used is printed at the debug level.
tf.get_logger().setLevel("DEBUG")

# Generate data.
x = np.random.rand(5000, 200, 750).astype(np.float32)
x += 0.01
x.clip(min=0, max=1, out=x)
y = np.random.randint(2, size=(5000, 1), dtype=np.int32)

def gru_cudnn(input_shape=(200, 750), dropout_rate=0.5):
    model = tfk.Sequential()
    model.add(tfkl.InputLayer(input_shape))
    model.add(tfkl.Masking(mask_value=0.0))
    model.add(tfkl.GRU(128))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.Dense(1))
    return model

# Train model (uses cuDNN implementation).
model = gru_cudnn()
model.compile(
    optimizer=tfk.optimizers.Adam(1e-3), 
    loss=tfk.losses.BinaryCrossentropy(from_logits=True))
# DEBUG:tensorflow:Layer gru will use cuDNN kernel when run on GPU.
model.fit(x, y)
# 157/157 [==============================] - 3s 20ms/step - loss: 0.8034

# Train using mirrored strategy, but using only one GPU.
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0"])
with strategy.scope():
    model = gru_cudnn()
    model.compile(
        optimizer=tfk.optimizers.Adam(1e-3), 
        loss=tfk.losses.BinaryCrossentropy(from_logits=True))
# DEBUG:tensorflow:Layer gru_1 will use cuDNN kernel when run on GPU.
model.fit(x, y)
# INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
# INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
# INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
# INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
# 157/157 [==============================] - 44s 279ms/step - loss: 0.8120

