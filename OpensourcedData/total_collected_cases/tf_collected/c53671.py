import tensorflow as tf
tf.debugging.set_log_device_placement(True)
        
_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
x = _input
with tf.device("/GPU:1"):
    x = tf.keras.layers.Dense(10, name="should_be_on_gpu")(x)
    x = tf.keras.layers.Dense(10, name="should_be_on_gpu_2")(x)
model = tf.keras.models.Model(inputs=[_input], outputs=[x])
model.compile('adam', 'mse')
model.summary()
model.fit([2], [4])

