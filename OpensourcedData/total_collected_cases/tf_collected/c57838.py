Conv2D layer with "same" padding fails to run with XLA on CUDA, though it works well with XLA on CPU.
I guess there's some issue with XLA for CUDA to compile Conv2D.



### Standalone code to reproduce the issue

https://colab.research.google.com/drive/1o9NX4ZzhyuhBSI5MFX-REgaBOxrCOpuM?usp=sharing

Reproduced on 2.8.2, 2.11.0.dev20220921

python
import tensorflow as tf
print(tf.__version__)
from keras import layers

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.conv = layers.Conv2D(
            filters=2, kernel_size=1, padding='same',
            dtype=tf.float32, autocast=False,
        )

    @tf.function(jit_compile=True)
    def __call__(self, x):
        y = self.conv(x)
        return y



inp = {
    "x": tf.constant(1.2, shape=[1,2,2,2], dtype=tf.float32),
}
m = MyModule()

with tf.device('CPU:0'):
    out = m(**inp)
    print(f'{out}')

with tf.device('GPU:0'):
    out = m(**inp) # <--- exception!
    print(f'{out}')

