import tensorflow as tf
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(2, 1, padding='valid', dtype=tf.float64, autocast=False)

    @tf.function(jit_compile=True) # without jit_compile=True works fine
    def __call__(self, i0):
        o0 = tf.floor(i0)
        o1 = self.conv(o0)
        o2 = tf.add(o1, o0)
        return o2

def simple():
    inp = {
        "i0": tf.constant(
            3.14, shape=[1,1,3,2], dtype=tf.float64
        ),
    }
    m = MyModule()

    out = m(**inp) # Error!

    print(out)
    print(out.shape)

if __name__ == "__main__":
    simple()

