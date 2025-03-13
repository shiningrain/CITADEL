conv = layers.Conv2D(1, 2, 1, autocast=False)
x = tf.random.normal([2, 1, 2, 2])
print(conv(x)) # no error


I think the behavior should be consistent in all modes.


### Standalone code to reproduce the issue

python
import tensorflow as tf
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        self.conv = layers.Conv2D(1, 2, 1, autocast=False)
    
    @tf.function
    def __call__(self, x):
        return self.conv(x)

if __name__ == '__main__':
    model = MyModule()

    tf.config.run_functions_eagerly(True)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # tf.Tensor([], shape=(2, 0, 1, 1), dtype=float32)

    tf.config.run_functions_eagerly(False)
    x = tf.random.normal([2, 1, 2, 2])
    print(model(x)) # Error when tracing  
    model.__call__.get_concrete_function(x) # Same error if we call this instead of the last line

