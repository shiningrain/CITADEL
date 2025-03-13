import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from absl.flags import FLAGS
from tensorflow import keras

flags.DEFINE_list("gpu",
                  default=None,
                  help="index of GPU")
flags.DEFINE_bool("recompute_grad",
                  default=False,
                  help="whether to recompute gradients to save GPU RAM")
flags.DEFINE_integer("batch_size",
                     default=1024,
                     help="batch size")
flags.DEFINE_bool("graph",
                  default=False,
                  help="use graph mode instead of eager mode")


def dense_lenet(inputs):
    net = keras.layers.Conv2D(32, 5, strides=2, use_bias=False, padding="SAME")(inputs)

    for _ in range(5):
        def _block(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(16, 1, use_bias=False, padding="SAME")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(4, 3, use_bias=False, padding="SAME")(x)
            return x
        if FLAGS.recompute_grad:
            _block = tf.recompute_grad(_block)
        net = keras.layers.concatenate([net, _block(net)])

    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)
    net = keras.layers.Conv2D(64, 1, use_bias=False, padding="SAME")(net)
    net = keras.layers.AveragePooling2D()(net)

    for _ in range(10):
        def _block(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(32, 1, use_bias=False, padding="SAME")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(8, 3, use_bias=False, padding="SAME")(x)
            return x
        if FLAGS.recompute_grad:
            _block = tf.recompute_grad(_block)
        net = keras.layers.concatenate([net, _block(net)])

    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)
    net = keras.layers.Conv2D(128, 1, use_bias=False, padding="SAME")(net)
    net = keras.layers.AveragePooling2D()(net)

    for _ in range(10):
        def _block(x):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(32, 1, use_bias=False, padding="SAME")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = keras.layers.Conv2D(8, 3, use_bias=False, padding="SAME")(x)
            return x
        if FLAGS.recompute_grad:
            _block = tf.recompute_grad(_block)
        net = keras.layers.concatenate([net, _block(net)])

    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)
    net = keras.layers.GlobalAveragePooling2D()(net)

    net = keras.layers.Dense(10)(net)
    net = keras.layers.Softmax()(net)

    return net


def main(_):
    if FLAGS.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, FLAGS.gpu))
    if FLAGS.graph:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.keras.backend.set_session(
            session=tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    gpu_options=tf.compat.v1.GPUOptions(
                        allow_growth=True
                    )
                )
            )
        )
    else:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    tfds.core.constants.DATA_DIR = "data"
    dataset_builder = tfds.image.FashionMNIST(version="3.*.*")
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset(
        split="train",
        shuffle_files=True,
        as_supervised=True,
    ).repeat().batch(FLAGS.batch_size)

    inputs = keras.layers.Input((28, 28, 1), batch_size=FLAGS.batch_size)
    model = keras.Model(inputs, dense_lenet(inputs))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        x=dataset,
        epochs=3,
        steps_per_epoch=60000//FLAGS.batch_size,
    )


if __name__ == "__main__":
    app.run(main)

