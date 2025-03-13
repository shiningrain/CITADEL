ConverterError: /usr/lib/python3.9/site-packages/tensorflow/python/ops/math_ops.py:5037:0: error: 'tf.Acos' op is neither a custom op nor a flex op
/usr/lib/python3.9/site-packages/tensorflow/python/util/dispatch.py:201:0: note: called from
/home/barabanus/work/4a-games/pfnn-tools/test/tflite-acos-missing.py:6:0: note: called from
/usr/lib/python3.9/site-packages/IPython/utils/py3compat.py:168:0: note: called from
/usr/lib/python3.9/site-packages/IPython/core/interactiveshell.py:2740:0: note: called from
/usr/lib/python3.9/site-packages/IPython/core/shellapp.py:377:0: note: called from
/usr/lib/python3.9/site-packages/IPython/core/shellapp.py:452:0: note: called from
/usr/lib/python3.9/site-packages/IPython/core/shellapp.py:328:0: note: called from
/usr/lib/python3.9/site-packages/IPython/terminal/ipapp.py:323:0: note: called from
/usr/lib/python3.9/site-packages/traitlets/config/application.py:87:0: note: called from
<unknown>:0: error: failed while converting: 'main': Ops that need custom implementation (enabled via setting the -emit-custom-ops flag):
	tf.Acos {device = ""}


**Standalone code to reproduce the issue** 

# use tensorflow API v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = tf.placeholder(dtype=tf.float32, shape=[], name="X")
Y = tf.acos(X, name="Y")

with tf.Session() as session:
    acosZero = session.run(Y, feed_dict={ X: 0., })
    acosOne = session.run(Y, feed_dict={ X: 1., })
    print(f"TensorFlow: acos(0.)={acosZero}, acos(1.)={acosOne}")

    converter = tf.lite.TFLiteConverter.from_session(
        session,
        input_tensors=[X],
        output_tensors=[Y]
    )
    flatbuffer = converter.convert()

