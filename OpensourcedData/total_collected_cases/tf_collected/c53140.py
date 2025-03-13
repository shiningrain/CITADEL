021-11-20 12:57:08.333476: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-20 12:57:09.302772: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15397 MB memory:  -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:3b:00.0, compute capability: 6.0
2021-11-20 12:57:09.303502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 15397 MB memory:  -> device: 1, name: Tesla P100-PCIE-16GB, pci bus id: 0000:b1:00.0, compute capability: 6.0
2021-11-20 12:57:10.556310: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at xla_ops.cc:241 : INVALID_ARGUMENT: Trying to access resource _AnonymousVar3 (defined @ /home/sdb/wda/tf_xla/lib/python3.7/site-packages/keras/engine/base_layer_utils.py:129) located in device /job:localhost/replica:0/task:0/device:GPU:1 from device /job:localhost/replica:0/task:0/device:GPU:0
Traceback (most recent call last):
  File "/home/sdb/wda/TF2-jit-compile-on-multi-gpu/xla_tf_function_distributed.py", line 59, in <module>
    train_step_dist(images, labels)
  File "/home/sdb/wda/tf_xla/lib/python3.7/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/sdb/wda/tf_xla/lib/python3.7/site-packages/tensorflow/python/eager/execute.py", line 59, in quick_execute
    inputs, attrs, num_outputs)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Trying to access resource _AnonymousVar3 (defined @ /home/sdb/wda/tf_xla/lib/python3.7/site-packages/keras/engine/base_layer_utils.py:129) located in device /job:localhost/replica:0/task:0/device:GPU:1 from device /job:localhost/replica:0/task:0/device:GPU:0 [Op:__inference_train_step_dist_650]


Describe the expected behavior
I want to use XLA  compile MirroredStrategy， because I found that _XLA can now compile MirroredStrategy: the step function passed to`strategy.run` can now be annoted with `jit_compile=True`._ from RELEASE.md from  2.5.0

Standalone code to reproduce the issue:


import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# Size of each input image, 28 x 28 pixels
IMAGE_SIZE = 28 * 28
# Number of distinct number labels, [0..9]
NUM_CLASSES = 10
# Number of examples in each training batch (step)
TRAIN_BATCH_SIZE = 100
# Number of training steps to run
TRAIN_STEPS = 1000

# Loads MNIST dataset.
train, test = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).repeat()


# Casting from raw data to the required datatypes.
def cast(images, labels):
    images = tf.cast(
        tf.reshape(images, [-1, IMAGE_SIZE]), tf.float32)
    labels = tf.cast(labels, tf.int64)
    return (images, labels)


layer = tf.keras.layers.Dense(NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam()


@tf.function(jit_compile=True)
def compiled_step(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
        predicted_labels = layer(images)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predicted_labels, labels=labels
        ))
    gradients = tape.gradient(loss, layer.trainable_variables)
    return loss, predicted_labels, gradients

@tf.function()
def train_step(images, labels):
    loss, pred, gradients = compiled_step(images, labels)
    optimizer.apply_gradients(zip(gradients, layer.trainable_variables))


strategy = tf.distribute.MirroredStrategy()

@tf.function(jit_compile=True)
def train_step_dist(image, labels):
    strategy.run(train_step, args=(image, labels))


for images, labels in train_ds:
    if optimizer.iterations > TRAIN_STEPS:
        break
    train_step_dist(images, labels)


