current behaviour:
when using mixed_bfloat16 policy, sigmoid type activation and dropout layer run on CPU, and LSTM layer does not support bfloat16 input.

expected behavior:
sigmoid/swish activation, dropout and LSTM layers run on GPU with mixed_bfloat16 policy



### Standalone code to reproduce the issue

shell
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import LSTM, Embedding

tf.debugging.set_log_device_placement(True)
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()


policy = mixed_precision.Policy('mixed_bfloat16')
# float32
print(policy.name, policy.variable_dtype, policy.compute_dtype)

mixed_precision.set_global_policy(policy)

input_shape = (4, 28, 28, 3)
x = tf.random.normal((4, 28, 28, 3))
#layer = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=(28, 28, 3))
layer1 = tf.keras.layers.Activation(activation=tf.keras.activations.swish)
layer2 = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)
#layer3 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)
layer4 = tf.keras.layers.Dropout(0.1, name="dropout")

x1 = layer1(x)
x2 = layer2(x1)
y = layer4(x2, training=True)
print(' ====== output ======== ', y.dtype, layer1.dtype, layer2.dtype, layer4.dtype)

vocab_size = 50
input_shape = (4, vocab_size, 1)
x = tf.random.normal(input_shape)

def customer_model():
    model = Sequential()
    model.add(LSTM(128))
    return model

model = customer_model()
model.build(input_shape=input_shape)
model.summary()
y = model(x)

