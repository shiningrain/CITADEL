❯ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.4 LTS
Release:        20.04
Codename:       focal

In [2]: tf.__version__
Out[2]: '2.11.0-dev20220914'


### 2. Code

python
import tensorflow as tf
from keras import layers

def get_tflite_callable(model, inp_dict):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        funcs=[model.__call__.get_concrete_function(**inp_dict)],
        trackable_obj=model,
    )
    tflite_bytes = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    runner = interpreter.get_signature_runner()
    return runner

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = layers.Dense(1)

    @tf.function
    def __call__(self, i0):
        o0 = tf.add(i0, tf.constant(1.1, shape=[1,1,1,1]))
        o0 = self.dense_1(o0) # a dense layer following an Add operator
        return o0


if __name__ == "__main__":
    inp = { "i0": tf.constant(0.2) }
    m = MyModule()
    runner = get_tflite_callable(m, inp)

    print(m(**inp)) # works fine
    
    print(runner(**inp)) # works fines

