import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from absl import app, flags
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("logs", "logs", "logs dir")
flags.DEFINE_integer("index", 0, "worker index")

class ThreeLayerMLP(keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = layers.Dense(32, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(16, activation='relu', name='dense_2')
        self.pred_layer = layers.Dense(
            1,
            activation='sigmoid',
            name='predictions',
        )

    def call(self, inputs, training=None):
        print(inputs.shape)
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)


def prepare_data():
    np.random.seed(0)
    x_train, y_train = (
        np.random.random((6000000, 31)),
        np.random.randint(2, size=(6000000, 1)),
    )

    x_val, y_val = (
        np.random.random((10000, 31)),
        np.random.randint(2, size=(10000, 1)),
    )

    return ((x_train, y_train), (x_val, y_val))


def main(argv):
    del argv  # Unused args
    tf_config = {
        "cluster": {
            "worker": ["ip1:12345", "ip2:12345"],
        },
        "task": {
            "index": FLAGS.index,
            "type": "worker"
        }
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print(json.loads(os.environ["TF_CONFIG"]))
    # distributed strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 128
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

    with strategy.scope():
        model = ThreeLayerMLP(name='3_layer_mlp')
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["AUC"],
        )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.logs,
        histogram_freq=10,
        update_freq=100,
    )

    ((x_train, y_train), (x_val, y_val)) = prepare_data()

    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[tensorboard_callback],
    )


if __name__ == '__main__':
    app.run(main)

