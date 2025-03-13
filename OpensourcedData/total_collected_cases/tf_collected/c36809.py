import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    x = tf.keras.backend.placeholder(shape=(), dtype=tf.float32)
    y = tf.keras.backend.placeholder(shape=(), dtype=tf.float32)
    training = tf.keras.backend.placeholder(shape=(), dtype=tf.bool)
    z = tf.cond(training, lambda: x * 10, lambda: y * 2)

graph_def = graph.as_graph_def()


def fn(*args):
    x_value, y_value, training_value = args
    return tf.graph_util.import_graph_def(graph_def,
                                          input_map={
                                              x.op.name: x_value,
                                              y.op.name: y_value,
                                              training.op.name: training_value,
                                          },
                                          return_elements=[z.name])


class ImportedCondTest(tf.test.TestCase):

    def test_in_graph_mode(self):
        with tf.Graph().as_default():
            out = tf.graph_util.import_graph_def(graph_def,
                                                 input_map={
                                                     x.op.name: 0.,
                                                     y.op.name: 1.0,
                                                     training.op.name: True,
                                                 },
                                                 return_elements=[z.name])
            with tf.compat.v1.Session() as sess:
                sess.run(out)

    def test_fn(self):
        fn(0., 1., True)

    def test_tf_function_fn(self):
        tf.function(fn)(0., 1., True)

    def test_map(self):
        dataset = tf.data.Dataset.from_tensor_slices(([0.], [1.], [True]))
        mapped = dataset.map(fn)
        for _ in mapped:
            pass


if __name__ == '__main__':
    tf.test.main()


**Other info / logs**

Running tests under Python 3.6.8: ...anaconda2/envs/tf-2.0-cpu/bin/python
[ RUN      ] ImportedCondTest.test_in_graph_mode
2020-02-17 15:13:40.567294: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-02-17 15:13:40.587856: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3491630000 Hz
2020-02-17 15:13:40.588472: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f5c5ffbd30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-02-17 15:13:40.588510: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[       OK ] ImportedCondTest.test_in_graph_mode
[ RUN      ] ImportedCondTest.test_map
2020-02-17 15:13:40.733678: W tensorflow/core/framework/op_kernel.cc:1655] OP_REQUIRES failed at optimize_dataset_op.cc:60 : Invalid argument: Unable to find FunctionDef for cond_true_4 in the registry.
[  FAILED  ] ImportedCondTest.test_map
[ RUN      ] ImportedCondTest.test_normal
[  FAILED  ] ImportedCondTest.test_normal
[ RUN      ] ImportedCondTest.test_session
[  SKIPPED ] ImportedCondTest.test_session
[ RUN      ] ImportedCondTest.test_tf_function
[  FAILED  ] ImportedCondTest.test_tf_function
======================================================================
ERROR: test_map (__main__.ImportedCondTest)
test_map (__main__.ImportedCondTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "cond_fail.py", line 47, in test_map
    for _ in mapped:
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 418, in __iter__
    return iterator_ops.OwnedIterator(self)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/data/ops/iterator_ops.py", line 594, in __init__
    self._create_iterator(dataset)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/data/ops/iterator_ops.py", line 600, in _create_iterator
    dataset = dataset._apply_options()
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 381, in _apply_options
    static_optimization_configs)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/data/ops/dataset_ops.py", line 4213, in __init__
    **self._flat_structure)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_dataset_ops.py", line 3638, in optimize_dataset
    _ops.raise_from_not_ok_status(e, name)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 6606, in raise_from_not_ok_status
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Unable to find FunctionDef for cond_true_4 in the registry. [Op:OptimizeDataset]

======================================================================
ERROR: test_normal (__main__.ImportedCondTest)
test_normal (__main__.ImportedCondTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "cond_fail.py", line 39, in test_normal
    fn(0., 1., True)
  File "cond_fail.py", line 21, in fn
    return_elements=[z.name])
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py", line 405, in import_graph_def
    producer_op_list=producer_op_list)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py", line 487, in _import_graph_def_internal
    validate_colocation_constraints)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py", line 221, in _PopulateTFImportGraphDefOptions
    dst_output = input_dst._as_tf_output()  # pylint: disable=protected-access
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1125, in _as_tf_output
    "_as_tf_output not supported when eager execution is enabled.")
NotImplementedError: _as_tf_output not supported when eager execution is enabled.

======================================================================
ERROR: test_tf_function (__main__.ImportedCondTest)
test_tf_function (__main__.ImportedCondTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "cond_fail.py", line 42, in test_tf_function
    tf.function(fn)(0., 1., True)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/def_function.py", line 568, in __call__
    result = self._call(*args, **kwds)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/def_function.py", line 638, in _call
    return self._concrete_stateful_fn._filtered_call(canon_args, canon_kwds)  # pylint: disable=protected-access
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1611, in _filtered_call
    self.captured_inputs)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 1692, in _call_flat
    ctx, args, cancellation_manager=cancellation_manager))
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/function.py", line 545, in call
    ctx=ctx)
  File "...anaconda2/envs/tf-2.0-cpu/lib/python3.6/site-packages/tensorflow_core/python/eager/execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.InvalidArgumentError: Op type not registered 'cond_true_4' in binary running on jackd-5810. Make sure the Op and Kernel are registered in the binary running in this process. Note that if you are loading a saved graph which used ops from tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before importing the graph, as contrib ops are lazily registered when the module is first accessed. while building NodeDef 'import/cond/then/_0' [Op:__inference_fn_61]

----------------------------------------------------------------------
Ran 5 tests in 0.243s

FAILED (errors=3, skipped=1)

