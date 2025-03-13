import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.embedding = tf.Variable(tf.random.uniform([50, 16]))

    def call(self, x):
        return tf.nn.embedding_lookup(self.embedding, x)

layer = MyLayer()

@tf.function
def _run(x):
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.math.reduce_sum(y)
    gradients = tape.gradient(loss, layer.weights)
    print("Gradient shape:", gradients[0].shape)

_run(tf.random.uniform([4, 16], minval=0, maxval=50, dtype=tf.int64))

