Traceback (most recent call last):
  File "/storage/tf_complex64_bug.py", line 67, in <module>
    model2.fit(data)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py", line 1178, in fit
    tmp_logs = self.train_function(iterator)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
    result = self._call(*args, **kwds)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 933, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 763, in _initialize
    self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 3050, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 3444, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 3279, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py", line 999, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 672, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/root/miniconda3/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py", line 986, in wrapper
    raise e.ag_error_metadata.to_exception(e)
TypeError: in user code:

    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:850 train_function  *
        return step_function(self, iterator)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:840 step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:1285 run
        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2833 call_for_each_replica
        return self._call_for_each_replica(fn, args, kwargs)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/mirrored_strategy.py:678 _call_for_each_replica
        return mirrored_run.call_for_each_replica(
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/mirrored_run.py:104 call_for_each_replica
        return _call_for_each_replica(strategy, fn, args, kwargs)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/mirrored_run.py:245 _call_for_each_replica
        coord.join(threads)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/training/coordinator.py:389 join
        six.reraise(*self._exc_info_to_raise)
    /root/miniconda3/lib/python3.8/site-packages/six.py:703 reraise
        raise value
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/training/coordinator.py:297 stop_on_exception
        yield
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/mirrored_run.py:238 _call_for_each_replica
        merge_result = threads[0].merge_fn(distribution, *merge_args,
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/keras/optimizer_v2/utils.py:148 _all_reduce_sum_fn  **
        return distribution.extended.batch_reduce_to(ds_reduce_util.ReduceOp.SUM,
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2402 batch_reduce_to
        return self._batch_reduce_to(reduce_op, value_destination_pairs, options)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/mirrored_strategy.py:767 _batch_reduce_to
        return cross_device_ops.batch_reduce(
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:446 batch_reduce
        return self.batch_reduce_implementation(reduce_op, value_destination_pairs,
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:874 batch_reduce_implementation
        return self._batch_all_reduce(reduce_op,
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:887 _batch_all_reduce
        dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:910 _do_batch_all_reduce
        device_grad_packs, tensor_packer = _pack_tensors(grouped, self._num_packs)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:820 _pack_tensors
        device_grad_packs = tensor_packer.pack(device_grads)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/distribute/cross_device_ops.py:747 pack
        concat_grads = array_ops.concat(flat_grads, 0)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
        return target(*args, **kwargs)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py:1768 concat
        return gen_array_ops.concat_v2(values=values, axis=axis, name=name)
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/ops/gen_array_ops.py:1227 concat_v2
        _, _, _op, _outputs = _op_def_library._apply_op_helper(
    /root/miniconda3/lib/python3.8/site-packages/tensorflow/python/framework/op_def_library.py:466 _apply_op_helper
        raise TypeError("%s that don't all match." % prefix)

    TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have types [float32, float32, float32, float32, complex64, complex64, float32, float32] that don't all match.


**Describe the expected behavior**
The model should train successfully

**[Contributing](https://www.tensorflow.org/community/contribute)**

- Do you want to contribute a PR? (yes/no):
- Briefly describe your candidate solution(if contributing):

**Standalone code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.


import tensorflow as tf
import numpy as np

def complex_uniform_initializer(scale=0.05):
    real_initializer = tf.keras.initializers.RandomUniform(-scale,scale)
    def initializer(shape,dtype):
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = real_initializer(shape,dtype)
        imag = real_initializer(shape,dtype)
        return tf.dtypes.complex(real,imag)
    return initializer

class ComplexDenseLayer(tf.keras.layers.Layer):

    def __init__(self, out_units, activation=None):
        super().__init__()
        self.out_units = out_units
        self.activation = activation

    def build(self, input_shape):
        inp_units = input_shape[-1]
        initializer = complex_uniform_initializer()
        self.w = self.add_weight(shape=[inp_units, self.out_units],
                                 initializer = initializer,
                                 dtype=tf.complex64, trainable=True)
        self.b = self.add_weight(shape=[self.out_units],
                                 initializer = initializer,
                                 dtype=tf.complex64, trainable=True)

    def call(self,inp):
        x = tf.einsum('bi,ij->bj', inp, self.w)
        x = tf.nn.bias_add(x, self.b)
        return self.activation(x)

    

def model(input_units, intermediate_units, output_units):
    inp = tf.keras.layers.Input((input_units,))
    xreal = tf.keras.layers.Dense(intermediate_units)(inp)
    ximag = tf.keras.layers.Dense(intermediate_units)(inp)
    x = tf.cast(xreal, 'complex64') + 1j*tf.cast(ximag,'complex64')
    x = ComplexDenseLayer(intermediate_units, activation = lambda w: w * tf.math.conj(w))(x)
    x = tf.math.real(x)
    x = tf.keras.layers.Dense(output_units)(x)
    return tf.keras.Model(inp,x) 

nsamples = 100
bsize = 10
ninp,nintermediate,nout = 16,128,16
inp = np.random.rand(nsamples, ninp)
tar = np.random.rand(nsamples, nout)
data = tf.data.Dataset.from_tensor_slices((inp,tar)).batch(bsize)

#Single GPU training works fine
model1 = model(ninp,nintermediate,nout)
model1.summary()
model1.compile(loss='mse', optimizer='adam')
model1.fit(data)

#Distributed training fails
distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    model2 = model(ninp,nintermediate,nout)
    model2.summary()
    model2.compile(loss='mse', optimizer='adam')
    model2.fit(data)

