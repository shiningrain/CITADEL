import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# logical device separation
physical_devices = tf.config.list_physical_devices('GPU')

if True:
  tf.config.set_logical_device_configuration( 
      physical_devices[0], 
      [tf.config.LogicalDeviceConfiguration(memory_limit=8000), 
       tf.config.LogicalDeviceConfiguration(memory_limit=8000)])


# my full batch dependent loss
def my_loss(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))

# my toy model
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
with mirrored_strategy.scope():
    model_distributed = Sequential(Dense(10))
    model_distributed.compile(loss=my_loss)

# my toy data
x = tf.random.normal([32, 10])
y = tf.random.normal([32, 10])

# my experiments
metrics = model_distributed.evaluate(x, y)
print(metrics)

y_pred = model_distributed.predict(x)
print(my_loss(y, y_pred))

