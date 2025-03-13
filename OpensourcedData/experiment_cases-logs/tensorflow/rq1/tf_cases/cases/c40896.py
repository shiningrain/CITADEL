import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CustomLayer, self).__init__(name=name, **kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, inputs):
        output_1 = self.conv_1(inputs)
        output_2 = self.conv_2(inputs)

        return output_1, output_2

    def compute_output_shape(self, input_shape):
        output_shape_1 = self.conv_1.compute_output_shape(input_shape)
        output_shape_2 = self.conv_2.compute_output_shape(input_shape)

        return output_shape_1, output_shape_2


if __name__ == "__main__":
    inputs = tf.keras.Input(shape=(None, None, None, 1))

    custom_layer = CustomLayer()
    output_1, output_2 = func_cls(custom_layer)(inputs)


