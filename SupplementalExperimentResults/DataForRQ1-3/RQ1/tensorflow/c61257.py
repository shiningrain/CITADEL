from tensorflow import keras
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])


def bilstm(num_units=25, input_shape=10):
# bilstm input layer
    input_tensor = keras.Input(shape=input_shape)
# bilstm hidden layer
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=8, embeddings_initializer="uniform")(input_tensor)
    x = func_cls(alpha=-0.2044550861511304)(x)
# bilstm output layer
    output_tensor = x
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    bilstm().summary()

