WARNING:tensorflow:Gradients do not exist for variables ['my_model/dense_1/kernel:0', 'my_model/dense_1/bias:0'] when minimizing the loss.
WARNING:tensorflow:Gradients do not exist for variables ['my_model/dense_1/kernel:0', 'my_model/dense_1/bias:0'] when minimizing the loss.

As a consequence, they never get updated during training.


**Describe the expected behavior**

Gradients for those tensors should be computed and they should get updated.

**Standalone code to reproduce the issue**

python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

num_classes = 500
num_epochs = 3
num_samples = 10000
batch_size = 10
learning_rate = 0.001

y = np.random.randint(0, num_classes, num_samples, dtype=np.int64)
x = np.expand_dims(y.astype(np.float32), -1)

x_test = x[:10]
y_test = y[:10]


class MyModel(Model):

    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = Dense(10)
        self.dense2 = Dense(num_classes)
        self.first_step = True

    def call(self, inputs, training=None, mask=None):
        hidden = self.dense1(inputs)
        if training and not self.first_step:
            return None, hidden
        else:
            logits = self.dense2(hidden)
            return logits, hidden


class SampledSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, decoder_obj=None, num_classes=0):
        super().__init__()
        self.decoder_obj = decoder_obj
        self.num_classes = num_classes

    def call(self, labels, hidden):
        labels = tf.cast(tf.expand_dims(labels, -1), tf.int64)

        weights = tf.transpose(self.decoder_obj.get_weights()[0])
        biases = self.decoder_obj.get_weights()[1]

        sampled_values = tf.random.uniform_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=5,
            range_max=self.num_classes,
            unique=False
        )

        loss_val = tf.nn.sampled_softmax_loss(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=hidden,
            num_sampled=5,
            num_classes=self.num_classes,
            sampled_values=sampled_values)

        return loss_val


my_model = MyModel(num_classes)
optimizer = SGD(learning_rate=learning_rate)
sampled_loss = SampledSoftmaxCrossEntropyLoss(
    decoder_obj=my_model.dense2, num_classes=num_classes)


def train_step(model, loss, optimizer, inputs, targets):
    with tf.GradientTape() as tape:
        logits, hidden = model(inputs, training=True)
        loss_val = loss(targets, hidden)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_val


def oredict(model, inputs):
    logits, _ = model(inputs, training=True)
    predictions = tf.argmax(logits, -1)
    return predictions


x_batches = np.split(x, 100)
y_batches = np.split(y, 100)

print(x_test)
print(oredict(my_model, x_test))

first_batch = True
for i in range(num_epochs):
    for x_batch, y_batch in zip(x_batches, y_batches):
        if first_batch:
            print("Weights and biases after first batch")
            print(my_model.dense2.get_weights()[0])
            print(my_model.dense2.get_weights()[1])
            first_batch = False

        loss_val = train_step(my_model, sampled_loss, optimizer, x_batch,
                              y_batch)
        print(loss_val)

print(x_test)
print(oredict(my_model, x_test))

print("Weights and biases after training")
print(my_model.dense2.get_weights()[0])
print(my_model.dense2.get_weights()[1])


