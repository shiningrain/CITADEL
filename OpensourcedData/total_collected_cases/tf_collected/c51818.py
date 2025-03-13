import time
import numpy as np
import tensorflow as tf

model0 = tf.keras.models.Sequential(
    tf.keras.layers.LSTM(128, input_shape=(300, 40))
)
model1 = tf.keras.models.Sequential(
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(128,))
)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam()

@tf.function
def train_step_0():
  with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
    f = model0(tf.random.normal([256, 300, 40]))
    y_p0 = model1(f)
    y_p1 = model1(tf.random.normal((256, 128)))
    loss0 = loss_object(tf.zeros_like(y_p0), y_p0)
    loss1 = loss_object(tf.ones_like(y_p1), y_p1)
  grad0 = tape0.gradient(loss0, model0.trainable_variables)
  grad1 = tape1.gradient(loss1, model1.trainable_variables)
  optimizer.apply_gradients(zip(grad0,model0.trainable_variables))
  optimizer.apply_gradients(zip(grad1,model1.trainable_variables))

t0=time.time()
for i in range(100):
  train_step_0()
print(time.time()-t0)

**output**

2021-09-03 12:39:22.262342: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:801] function_optimizer failed: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
2021-09-03 12:39:22.287055: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:801] function_optimizer failed: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
2021-09-03 12:39:22.300477: W tensorflow/core/common_runtime/process_function_library_runtime.cc:841] Ignoring multi-device function optimization failure: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
17.58007049560547

python
import time
import numpy as np
import tensorflow as tf

model0 = tf.keras.models.Sequential(
    tf.keras.layers.LSTM(128, input_shape=(300, 40))
)
model1 = tf.keras.models.Sequential(
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(128,))
)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam()

@tf.function
def train_step_1():
  with tf.GradientTape(persistent=True) as tape:
    f = model0(tf.random.normal([256, 300, 40]))
    y_p0 = model1(f)
    y_p1 = model1(tf.random.normal((256, 128)))
    loss0 = loss_object(tf.zeros_like(y_p0), y_p0)
    loss1 = loss_object(tf.ones_like(y_p1), y_p1)
  grad0 = tape.gradient(loss0, model0.trainable_variables)
  grad1 = tape.gradient(loss1, model1.trainable_variables)
  optimizer.apply_gradients(zip(grad0,model0.trainable_variables))
  optimizer.apply_gradients(zip(grad1,model1.trainable_variables))

t0=time.time()
for i in range(100):
  train_step_1()
print(time.time()-t0)

**output**

2021-09-03 12:43:44.947280: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:801] function_optimizer failed: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
2021-09-03 12:43:44.972180: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:801] function_optimizer failed: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
2021-09-03 12:43:44.985573: W tensorflow/core/common_runtime/process_function_library_runtime.cc:841] Ignoring multi-device function optimization failure: Invalid argument: Input 0 of node zeros_like_1 was passed float from sequential/lstm/PartitionedCall:6 incompatible with expected variant.
16.632988929748535

python
import time
import numpy as np
import tensorflow as tf

model0 = tf.keras.models.Sequential(
    tf.keras.layers.LSTM(128, input_shape=(300, 40))
)
model1 = tf.keras.models.Sequential(
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(128,))
)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam()

@tf.function
def train_step_2():
  with tf.GradientTape() as tape0:
    f = model0(tf.random.normal([256, 300, 40]))
    y_p0 = model1(f)
    loss0 = loss_object(tf.zeros_like(y_p0), y_p0)
  with tf.GradientTape() as tape1:
    y_p1 = model1(tf.random.normal((256, 128)))
    loss1 = loss_object(tf.ones_like(y_p1), y_p1)
  grad0 = tape0.gradient(loss0, model0.trainable_variables)
  grad1 = tape1.gradient(loss1, model1.trainable_variables)
  optimizer.apply_gradients(zip(grad0,model0.trainable_variables))
  optimizer.apply_gradients(zip(grad1,model1.trainable_variables))

t0=time.time()
for i in range(100):
  train_step_2()
print(time.time()-t0)

