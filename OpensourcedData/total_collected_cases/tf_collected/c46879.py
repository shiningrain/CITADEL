import tensorflow as tf

m = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest"),
    ]
)

converter = tf.lite.TFLiteConverter.from_keras_model(m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open("tmp-model.tflite", "wb") as f:
    f.write(converter.convert())

