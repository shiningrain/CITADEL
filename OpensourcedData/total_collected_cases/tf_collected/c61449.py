import tensorflow as tf
.....
checkpoint_dir = "s3://xxx/xx/"
tf.compat.v1.train.MonitoredTrainingSession(...., checkpoint_dir=checkpoint_dir, ...)

`checkpoint_dir` contains everything needed to restore variables, including checkpoint, graph.pbtxt, etc. Everything works fine.

After switching to TensorFlow 2.7.0, we realized that the Modular File System has been introduced into TensorFlow. So, we installed TensorFlow-io version 0.23.0, which is compatible with TensorFlow 2.7.0. The code becomes:
python
import tensorflow as tf
import tensorflow_io as tfio
.....
checkpoint_dir = "s3://xxx/xx/"
tf.compat.v1.train.MonitoredTrainingSession(...., checkpoint_dir=checkpoint_dir, ...)

However, it no longer works, and an error is reported:

.....
2023-08-02 16:02:40.147093: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at save_restore_v2_ops.cc:207 : DATA_LOSS: truncated block read
Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1380, in _do_call
    return fn(*args)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1364, in _run_fn
    target_list, run_metadata)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1458, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.DataLossError: 2 root error(s) found.
  (0) DATA_LOSS: truncated block read
         [[{{node save/RestoreV2}}]]
         [[save/RestoreV2/_1]]
  (1) DATA_LOSS: truncated block read
         [[{{node save/RestoreV2}}]]
0 successful operations.
0 derived errors ignored.
.....


## 2. Reproduce the issue using simple code
To rule out the possibility that the issue is caused by the complexity of the model in my project, I reproduced it using a very simple code.
### 2.1 Step 1: Train the model
First, I used the following code to train a very simple model and save it in a local directory:
python
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="x")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")

W = tf.Variable(tf.zeros([1, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

y_pred = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.square(y - y_pred))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)

global_step = tf.compat.v1.train.get_or_create_global_step()
train_op = optimizer.minimize(loss, global_step=global_step)

x_train = [[1], [2], [3], [4]]
y_train = [[0], [-1], [-2], [-3]]

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
hooks = [tf.compat.v1.train.StopAtStepHook(last_step=500)]

checkpoint_dir = './checkpoints'

with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                                 config=config,
                                                 hooks=hooks) as sess:
    while not sess.should_stop():
        sess.run(train_op, feed_dict={x: x_train, y: y_train})

### 2.2 Step 2: Upload the model to S3
Then, I used S3 tools to upload all materials in `./checkpoints` to a remote S3 path:

s3cmd put ./checkpoints/ s3://xxxx/xxx/checkpoints/

### 2.3 Step 3: Restore the model from S3 (error)
Finally, I restored the model training using the following code, and an error was reported:
python
import tensorflow as tf
import tensorflow_io as tfio

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="x")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")

W = tf.Variable(tf.zeros([1, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

y_pred = tf.matmul(x, W) + b
loss = tf.reduce_mean(tf.square(y - y_pred))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)

global_step = tf.compat.v1.train.get_or_create_global_step()

train_op = optimizer.minimize(loss, global_step=global_step)

x_train = [[1], [2], [3], [4]]
y_train = [[0], [-1], [-2], [-3]]

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

checkpoint_dir = 's3://xxxx/xxx/checkpoints/'

hooks = [tf.compat.v1.train.StopAtStepHook(last_step=2000)]
with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as sess:
    while not sess.should_stop():
        sess.run(train_op, feed_dict={x: x_train, y: y_train})

The full log is shown below:


WARNING:tensorflow:From /root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/training_util.py:401: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
2023-08-02 16:43:08.483327: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-08-02 16:43:09.090602: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38415 MB memory:  -> device: 0, name: A100-SXM4-40GB, pci bus id: 0000:0e:00.0, compute capability: 8.0
2023-08-02 16:43:09.854875: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at save_restore_v2_ops.cc:207 : DATA_LOSS: truncated block read
Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1380, in _do_call
    return fn(*args)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1364, in _run_fn
    target_list, run_metadata)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1458, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.DataLossError: 2 root error(s) found.
  (0) DATA_LOSS: truncated block read
         [[{{node save/RestoreV2}}]]
         [[save/RestoreV2/_1]]
  (1) DATA_LOSS: truncated block read
         [[{{node save/RestoreV2}}]]
0 successful operations.
0 derived errors ignored.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "train_s3.py", line 36, in <module>
    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as sess:
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 616, in MonitoredTrainingSession
    stop_grace_period_secs=stop_grace_period_secs)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1062, in __init__
    stop_grace_period_secs=stop_grace_period_secs)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 761, in __init__
    self._sess = _RecoverableSession(self._coordinated_creator)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1267, in __init__
    _WrappedSession.__init__(self, self._create_session())
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1272, in _create_session
    return self._sess_creator.create_session()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 914, in create_session
    self.tf_sess = self._session_creator.create_session()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 681, in create_session
    init_fn=self._scaffold.init_fn)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/session_manager.py", line 321, in prepare_session
    config=config)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/session_manager.py", line 251, in _restore_checkpoint
    sess, saver, ckpt.model_checkpoint_path)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/session_manager.py", line 71, in _restore_checkpoint_and_maybe_run_saved_model_initializers
    saver.restore(sess, path)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 1405, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 971, in run
    run_metadata_ptr)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1194, in _run
    feed_dict_tensor, options, run_metadata)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1374, in _do_run
    run_metadata)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1399, in _do_call
    raise type(e)(node_def, op, message)  # pylint: disable=no-value-for-parameter
tensorflow.python.framework.errors_impl.DataLossError: 2 root error(s) found.
  (0) DATA_LOSS: truncated block read
         [[node save/RestoreV2
 (defined at train_s3.py:36)
]]
         [[save/RestoreV2/_1]]
  (1) DATA_LOSS: truncated block read
         [[node save/RestoreV2
 (defined at train_s3.py:36)
]]
0 successful operations.
0 derived errors ignored.

Errors may have originated from an input operation.
Input Source operations connected to node save/RestoreV2:
In[0] save/Const:
In[1] save/RestoreV2/tensor_names:
In[2] save/RestoreV2/shape_and_slices:

Operation defined at: (most recent call last)
>>>   File "train_s3.py", line 36, in <module>
>>>     with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as sess:
>>> 

Input Source operations connected to node save/RestoreV2:
In[0] save/Const:
In[1] save/RestoreV2/tensor_names:
In[2] save/RestoreV2/shape_and_slices:

Operation defined at: (most recent call last)
>>>   File "train_s3.py", line 36, in <module>
>>>     with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as sess:
>>> 

Original stack trace for 'save/RestoreV2':
  File "train_s3.py", line 36, in <module>
    with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, config=config, hooks=hooks) as sess:
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 616, in MonitoredTrainingSession
    stop_grace_period_secs=stop_grace_period_secs)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1062, in __init__
    stop_grace_period_secs=stop_grace_period_secs)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 761, in __init__
    self._sess = _RecoverableSession(self._coordinated_creator)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1267, in __init__
    _WrappedSession.__init__(self, self._create_session())
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 1272, in _create_session
    return self._sess_creator.create_session()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 914, in create_session
    self.tf_sess = self._session_creator.create_session()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 672, in create_session
    self._scaffold.finalize()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/monitored_session.py", line 236, in finalize
    self._saver = training_saver._get_saver_or_default()  # pylint: disable=protected-access
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 625, in _get_saver_or_default
    saver = Saver(sharded=True, allow_empty=True)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 923, in __init__
    self.build()
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 935, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 973, in _build
    build_restore=build_restore)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 528, in _build_internal
    restore_sequentially, reshape)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 407, in _AddShardedRestoreOps
    name="restore_shard"))
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 354, in _AddRestoreOps
    restore_sequentially)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/training/saver.py", line 601, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/ops/gen_io_ops.py", line 1504, in restore_v2
    name=name)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py", line 746, in _apply_op_helper
    attrs=attr_protos, op_def=op_def)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py", line 3705, in _create_op_internal
    op_def=op_def)
  File "/root/miniconda3/lib/python3.7/site-packages/tensorflow/python/framework/ops.py", line 2101, in __init__
    self._traceback = tf_stack.extract_stack_for_node(self._c_op)



## 3. Test tf.io and s3 connectivity
I also use the following code to test if tf.io can access s3

import tensorflow as tf
import tensorflow_io as tfio
s3_path = "s3://xxxxx/xxx/checkpoints/checkpoint"
ret = tf.io.read_file(s3_path)
print(ret)

