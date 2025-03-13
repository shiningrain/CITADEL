Is it possible to integrate PocketFFT when using a CPU into Tensorflow, which functions like tf.signal.stft and tf.signal.inverse_stft can leverage? 

Currently, Tensorflow's FFT uses EigenFFT, which is almost 3x slower than Numpy and Jax, which use PocketFFT.

There are heaps more details here: https://github.com/tensorflow/tensorflow/issues/6541

I'm sure many projects would benefit from this investment considering much of what's done for speech and music these days use STFT data.



### Standalone code to reproduce the issue

shell
print("seconds (lower is better):")
print(f"Tensorflow {tf.__version__}", timeit.timeit('X = tf.signal.rfft(x)', setup='import tensorflow as tf; x = tf.random.normal([50000, 512])', number=10))
print(f"Tensorflow {tf.__version__}, double precision", timeit.timeit('X = tf.cast(tf.signal.rfft(tf.cast(x, tf.float64)), tf.complex64)', setup='import tensorflow as tf; x = tf.random.normal([50000, 512])', number=10))
print("Numpy: ", timeit.timeit('X = numpy.fft.rfft(x)', setup='import numpy.fft; import tensorflow as tf; x = tf.random.normal([50000, 512])', number=10))
print("Jax: ", timeit.timeit('jnp.fft.rfft(x).block_until_ready()', setup='import jax.numpy as jnp; import tensorflow as tf; x = tf.random.normal([50000, 512]).numpy()', number=10))

