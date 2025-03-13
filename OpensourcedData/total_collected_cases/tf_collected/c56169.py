The attached code should produce an output that is equal to the input.  All it does is take the input vector, split it, and reconcatenate it.



### Standalone code to reproduce the issue

shell
import tensorflow as tf
from tensorflow.keras import layers, models

test_input_data = tf.expand_dims(tf.linspace(0.0, 1.0, 6), axis=0)

input_layer = layers.Input(batch_shape=(1, 6))

outputs = []
for i_channel in range(6):
    print(f'Initialising layer that should extract element #{i_channel} - (expecting {test_input_data[0, i_channel]})')
    outputs.append(layers.Lambda(lambda c: c[..., i_channel])(input_layer))
cat_outputs = layers.Concatenate()(outputs)

model = models.Model(inputs=input_layer, outputs=cat_outputs, name='repro_model')

y = model(test_input_data)

print(y)

