import tensorflow as tf

class MyModel(tf.keras.layers.Layer):

    def __init__(self):
        super(MyModel, self).__init__()
        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(512) for _ in range(12)])
        self.rnn = tf.keras.layers.RNN(cell)

    def call(self, inputs):
        return self.rnn(inputs)

dataset = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([10000], minval=1, maxval=80, dtype=tf.int32))
dataset = dataset.shuffle(10000)
dataset = dataset.map(lambda t: tf.zeros([t, 512]))
dataset = dataset.padded_batch(
    64, padded_shapes=tf.compat.v1.data.get_output_shapes(dataset))
dataset = dataset.repeat()
dataset = dataset.prefetch(1)

devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
strategy = tf.distribute.MirroredStrategy(devices=devices)

with strategy.scope():
    dataset = strategy.experimental_distribute_dataset(dataset)
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam()

def step(inputs):
    outputs = model(inputs)
    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(
        tf.zeros_like(outputs), outputs)
    variables = model.trainable_variables
    gradients = optimizer.get_gradients(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

@tf.function
def train():
    with strategy.scope():
        for inputs in dataset:
            loss = strategy.experimental_run_v2(step, args=(inputs,))

train()

