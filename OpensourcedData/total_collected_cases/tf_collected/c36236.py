import tensorflow as tf

inputs = tf.keras.layers.Input(shape=[None, 1], dtype=tf.float32)
hidden = tf.keras.layers.GRU(10)(inputs)
hidden = tf.gather(hidden, [0])
output = tf.keras.layers.Dense(1)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=output)

@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.losses.mean_squared_error(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

train(tf.constant([[[1], [2], [3]]], dtype=tf.float32), tf.constant([[1]], dtype=tf.float32))

