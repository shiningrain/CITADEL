tensorflow.python.framework.errors_impl.InternalError: 2 root error(s) found.
  (0) Internal:  unhandled cuda error
         [[node Adam/NcclAllReduce (defined at workspace/gpu_tests/test_gpus.py:60) ]]
  (1) Internal:  unhandled cuda error
         [[node Adam/NcclAllReduce (defined at workspace/gpu_tests/test_gpus.py:60) ]]
         [[GroupCrossDeviceControlEdges_0/Adam/Adam/update_1_1/Const/_39]]
0 successful operations.
1 derived errors ignored. [Op:__inference_distributed_function_36247]

Function call stack:
distributed_function -> distributed_function

When I use cross_device_ops=tf.distribute.ReductionToOneDevice() it doesn't crash but it's not the optimal performance since it's not using NCCL. The NCCL seems to work however. Check the NCCL/all_reduce_perf log below. 

**Describe the expected behavior**
Training should not crash. 

**Code to reproduce the issue**
bash
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import tensorflow_datasets as tfds

LENGTH_DATASET = 17509
NUM_CLASSES = 9
IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 32


def mymap_func(features):
    return features["image"], features["label"]


AUTOTUNE = tf.data.experimental.AUTOTUNE

# create input pipeline
dataset = tfds.load(name="deep_weeds", split="train")
dataset = dataset.map(mymap_func,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=LENGTH_DATASET, seed=42,
                          reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True).repeat()
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# create model
img_width, img_height = 270, 270

shape, classes = (img_width, img_height, 1), 3

# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
strategy = tf.distribute.MirroredStrategy()
print("Number of devices in strategy: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():

    model = ResNet50(include_top=True,
                       weights=None,
                       input_tensor=None,
                       input_shape=IMG_SHAPE,
                       pooling=None,
                       classes=NUM_CLASSES)

    model.compile(optimizer=tf.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=["accuracy"])

    train_steps = np.ceil(LENGTH_DATASET / BATCH_SIZE)
    history = model.fit(
            x=dataset,
            epochs=10,
            verbose=1,
            steps_per_epoch=train_steps,
            use_multiprocessing=False,
            workers=8)

