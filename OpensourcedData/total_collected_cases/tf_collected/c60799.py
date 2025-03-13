import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time


# Set up the environment to use CPU
tf.config.set_visible_devices([], 'GPU')


#######################
#   CircularBuffer
#######################
class CircularBufferLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, buffer_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.buffer_size = buffer_size
        self.stride = stride
        self.buffer = self.add_weight(name='buffer', shape=(1, buffer_size, self.num_features),
                                      initializer='zeros', trainable=False, dtype=tf.float32)
        self.call_count = self.add_weight(name='call_count', shape=(), initializer='zeros',
                                          dtype=tf.int32, trainable=False)
        # total count, this count will never reset
        self.total_call_count = self.add_weight(name='total_call_count', shape=(), initializer='zeros',
                                                dtype=tf.int32, trainable=False)

    def call(self, inputs, **kwargs):
        # inputs should be reshaped to (1, 1, num_features) to match the buffer shape
        inputs = tf.reshape(inputs, [1, 1, self.num_features])

        # Update the buffer with the new data
        self.buffer.assign(tf.concat([self.buffer[:, 1:], inputs], axis=1))

        # Update the call count
        self.call_count.assign(tf.minimum(self.call_count + 1, self.stride))
        # self.total_call_count.assign(self.total_call_count + 1)
        self.total_call_count.assign(tf.minimum(self.total_call_count + 1, self.buffer_size))

        # If-else condition
        self.call_count.assign(
            tf.cond(tf.logical_and(tf.equal(self.call_count, self.stride), tf.greater_equal(self.total_call_count, self.buffer_size)),
                    true_fn=lambda: 0,
                    false_fn=lambda: self.call_count,
                    )
        )

        # Create a boolean flag indicating if self.call_count is 0
        # and the total number of calls to this layer is at least self.buffer_size
        flag = tf.equal(self.call_count, 0)

        # Return the buffer data and the flag
        return [self.buffer, flag]

    def reset(self):
        self.buffer.assign(tf.zeros_like(self.buffer))
        self.call_count.assign(tf.zeros_like(self.call_count))
        self.total_call_count.assign(tf.zeros_like(self.total_call_count))

    def get_config(self):
        config = {
            'buffer': self.buffer,
            'total_call_count': self.total_call_count,
            'call_count': self.call_count
        }
        base_config = super(CircularBufferLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#######################
#   Streaming Model
#######################
class StreamingModel(tf.keras.Model):
    def __init__(self, input_channel, output_channel, kernel_size, stride, **kwargs):
        super().__init__(**kwargs)

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride

        # set acoustic model buffer
        self.buffer = CircularBufferLayer(
            num_features=input_channel,
            buffer_size=kernel_size,
            stride=stride
        )

        # set filters
        self.conv1d = tf.keras.layers.Conv1D(
            output_channel,
            kernel_size=kernel_size,
            strides=1,
            use_bias=False,
            padding='valid',
            data_format='channels_last'
        )

    def call(self, inputs, **kwargs):
        # buffer:
        [x, flag] = self.buffer(inputs)              # output shape = [1, kernel_size, input_channel]

        x = tf.cond(flag,                       # output shape = [1, 1, output_channel]
            true_fn=lambda: self.conv1d(x),
            false_fn=lambda: tf.zeros([1, 1, self.output_channel])
        )

        x = tf.reshape(x, [1, self.output_channel])     # output shape = [1, output_channel]

        return [x, flag]

    def reset(self):
        self.buffer.reset()

if __name__ == '__main__':
    model = StreamingModel(
        input_channel=32,
        output_channel=64,
        kernel_size=5,
        stride=2
    )

    # create some dummy data with the correct shapes
    seq_len = 50
    input_data = tf.random.normal([1, seq_len, model.input_channel])

    # call the model on the dummy data - crucial to build all sub-graphs
    for t in range(seq_len):
        output = model(input_data[:, t])

    # reset the buffer after executing
    model.reset()

    # print the model's summary weights
    model.summary()

    #####################################
    #        TFLite Conversion
    #####################################
    tflite_path = '/data/netapp2/git-repos/danielr/sbu_whispro_tflm/tflite_models/open_issue_8-16'
    saved_model_dir = os.path.expanduser(tflite_path)
    tf.saved_model.save(obj=model, export_dir=saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.experimental_enable_resource_variables = True

    # Quantize the model to 16x8
    def representative_data_gen():
        input_channel = 32
        seq_len = 50
        for t in range(seq_len):
            x = tf.random.normal([1, input_channel])
            yield [x]

    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter.representative_dataset = representative_data_gen
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Save the TFLite model.
    with tf.io.gfile.GFile(tflite_path + '.tflite', 'wb') as f:
        f.write(tflite_model)

