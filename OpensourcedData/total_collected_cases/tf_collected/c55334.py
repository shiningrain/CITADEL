from PIL import Image

import tensorflow as tf
import numpy as np


class MyModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.applications.ResNet101(
          include_top=True,
          weights="imagenet",
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
      )

    def call(self, x):
        x = self.preprocess(x)
        return self.model(x)

    def preprocess(self, x):
        x = tf.map_fn(
            self.decode_bytes,
            x,
            dtype=tf.float32
        )
        x = tf.image.resize(x, (224, 224))

        return x

    def decode_bytes(self, x):
        return tf.io.decode_image(x, channels=3, dtype=tf.float32)


def create_input_tensor():
    fname = "/tmp/image.png"

    mock_input = np.random.random((330, 330, 3)).astype(np.float32)
    mock_image = Image.fromarray(mock_input, "RGB")
    mock_image.save(fname)
    b = tf.io.read_file(fname)
    return tf.expand_dims(b, axis=0)


model = MyModel()
in_tensor = create_input_tensor()

model.predict(in_tensor)

