import tensorflow as tf, cupy as cp

# initialize tf
x = tf.random.uniform((1,))

tf.experimental.dlpack.from_dlpack(cp.asfortranarray(cp.ones((5, 2))).toDlpack())


Output should be

InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-4-cb770ae77ab3> in <module>()
      1 import tensorflow as tf, cupy as cp
      2 x = tf.random.uniform((5,))
----> 3 tf.experimental.dlpack.from_dlpack(cp.asfortranarray(cp.ones((5, 2))).toDlpack())

/usr/local/lib/python3.6/dist-packages/tensorflow/python/dlpack/dlpack.py in from_dlpack(dlcapsule)
     64     A Tensorflow eager tensor
     65   """
---> 66   return pywrap_tfe.TFE_FromDlpackCapsule(dlcapsule, context.context()._handle)

InvalidArgumentError: Invalid strides array from DLPack

