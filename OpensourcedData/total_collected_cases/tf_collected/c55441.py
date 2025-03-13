import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.framework import ops

tf.get_logger().setLevel('WARNING')


def extract_devices_from_graphdef(graphdef):
    all_nodes = [n for n in graphdef.node]
    all_devices = list(set([n.device for n in all_nodes]))
    return all_devices


def create_model():
  """Define a simple sequential model"""
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])
  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
  return model


if __name__ == "__main__":

    try:
        gpus = tf.config.list_physical_devices('GPU')
    except AttributeError:
        gpus = tf.config.experimental.list_physical_devices('GPU')

    if not gpus:
        raise RuntimeError("No GPUs has been found.")

    print('Found the following GPUs:')
    for gpu in gpus:
        print(f"\t- {gpu}")


    # Create a basic model instance
    model = create_model()
    model.save('./saved_model/my_model')

    # Case 1 - Working
    with tf.device("gpu:1"):
        print("\n=================== CASE 1: `tf.saved_model.load` ===================")
        from tensorflow.python.saved_model import load as load_module
        from tensorflow.python.saved_model.load import load as load_fn
        print("TF2 API:      ", id(tf.saved_model.load))
        # >>> TF2 API:       139634355265248
        print("Direct Access:", id(load_fn))
        # >>> Direct Access: 139634355265248
        print("Module Access:", id(load_module.load))
        # >>> Module Access: 139634355265248

        model_loaded = tf.saved_model.load(export_dir='./saved_model/my_model')
        print("Loaded Model:", model_loaded.variables[0].device)
        # >>> '/job:localhost/replica:0/task:0/device:GPU:1'

        from tensorflow.python.saved_model import signature_constants
        func = model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

        print("Loaded Func Found Devices:", extract_devices_from_graphdef(func.graph.as_graph_def()))
        # >>> Loaded Func Found Devices: {''}

        from tensorflow.python.framework import convert_to_constants
        frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
        print("Frozen Func Found Devices:", extract_devices_from_graphdef(frozen_func.graph.as_graph_def()))

        print("\n=================== CASE 2: `loader.load` ===================")
        from tensorflow.python.saved_model import loader
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.client import session
        from tensorflow.python.saved_model import signature_constants

        with session.Session() as sess:
            input_meta_graph_def = loader.load(
                sess, [tag_constants.SERVING], './saved_model/my_model'
            )
            # input_signature_def = input_meta_graph_def.signature_def[
            #     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            # ]

            print("Found Devices:", extract_devices_from_graphdef(sess.graph.as_graph_def()))
            # >>> Found Devices: ['', '/device:CPU:0']

        print("\n=================== CASE 3: `importer.import_graph_def` ===================")
        from tensorflow.python.framework import importer

        print("Direct Access:", id(tf.graph_util.import_graph_def))
        # >>> Direct Access: 139634355265248
        print("Module Access:", id(importer.import_graph_def))
        # >>> Module Access: 139634355265248
        
        with ops.Graph().as_default() as graph:
            importer.import_graph_def(input_meta_graph_def.graph_def, name="")
            print("Found Devices:", extract_devices_from_graphdef(graph.as_graph_def()))
            # >>> Found Devices: ['', '/device:CPU:0']

