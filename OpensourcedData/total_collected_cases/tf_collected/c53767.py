import tensorflow as tf


class QuantConv2DTransposed(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", [3, 3, input_shape[-1], 24])

    def call(self, inputs):
        filters = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            self.kernel, -3.0 * tf.ones([24]), 3.0 * tf.ones([24]), narrow_range=True
        )
        filters = tf.transpose(filters, (0, 1, 3, 2))
        return tf.nn.conv2d_transpose(inputs, filters, [*inputs.shape[:-1], 24], 1)


inp = tf.keras.Input(shape=(6, 8, 48), batch_size=1)
x = tf.quantization.fake_quant_with_min_max_vars(inp, -3.0, 3.0, narrow_range=True)
x = QuantConv2DTransposed()(x)
x = tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0, narrow_range=True)

model = tf.keras.Model(inp, x)

model.save("/tmp/testing")
converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/testing")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# terminated by signal SIGSEGV (Address boundary error)
tflite_model = converter.convert()

