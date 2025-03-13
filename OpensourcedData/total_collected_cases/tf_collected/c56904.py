2022-07-26 14:10:05.665573: W tensorflow/core/grappler/costs/op_level_cost_estimator.cc:690] Error in PredictCost() for the op: op: "CropAndResize" attr { key: "T" value { type: DT_UINT8 } } attr { key: "extrapolation_value" value { f: 0 } } attr { key: "method" value { s: "bilinear" } } inputs { dtype: DT_UINT8 shape { dim { size: 1 } dim { size: 40 } dim { size: 40 } dim { size: 3 } } } inputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 4 } } } inputs { dtype: DT_INT32 shape { dim { size: -2 } } } inputs { dtype: DT_INT32 shape { dim { size: 2 } } value { dtype: DT_INT32 tensor_shape { dim { size: 2 } } int_val: 16 } } device { type: "CPU" vendor: "GenuineIntel" model: "101" frequency: 2500 num_cores: 10 environment { key: "cpu_instruction_set" value: "AVX SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2" } environment { key: "eigen" value: "3.4.90" } l1_cache_size: 32768 l2_cache_size: 1048576 l3_cache_size: 28835840 memory_size: 268435456 } outputs { dtype: DT_FLOAT shape { dim { size: -2 } dim { size: 16 } dim { size: 16 } dim { size: 3 } } }
2022-07-26 14:10:05.815003: W tensorflow/core/kernels/data/cache_dataset_ops.cc:856] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.


At first, I didn't care because my code was running fine, but I had a CPU memory error after about 9 epochs of ImageNet training. This suggests that there is somehow a memory leak during training.

The same behaviour (although on GPU from that I understand) was also observed in this [SO question](https://stackoverflow.com/q/72642906/4332585).



### Standalone code to reproduce the issue


Unfortunately, the warning does not appear on Colab, but here is a link with the appropriate minimal example anyway: https://colab.research.google.com/drive/1QHa4kxPLfCjkfDvmwv9wj5yq19za5SpA?usp=sharing

However, locally (on my laptop without GPU) and on my server, the warning is thrown.

The full code is the following:

python

import tensorflow as tf

class RandomResizedCrop(tf.keras.layers.Layer):
    # taken from
    # https://keras.io/examples/vision/nnclr/#random-resized-crops
    def __init__(self, scale, ratio, crop_shape):
        super(RandomResizedCrop, self).__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
        self.crop_shape = crop_shape

    def call(self, images):
        batch_size = tf.shape(images)[0]

        random_scales = tf.random.uniform(
            (batch_size,),
            self.scale[0],
            self.scale[1]
        )
        random_ratios = tf.exp(tf.random.uniform(
            (batch_size,),
            self.log_ratio[0],
            self.log_ratio[1]
        ))

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios),
            0,
            1,
        )
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios),
            0,
            1,
        )
        height_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_heights,
        )
        width_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_widths,
        )

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images,
            bounding_boxes,
            tf.range(batch_size),
            self.crop_shape,
        )
        return images

import tensorflow_datasets as tfds


ds = tfds.load('cifar10', split='train', as_supervised=True)
image_width = 16
crop = RandomResizedCrop(
    scale=(0.08, 1.0),
    ratio=(0.75, 1.33),
    crop_shape=(image_width, image_width),
)
data_aug_list = [
    tf.keras.layers.ZeroPadding2D(padding=4),
    crop,
]
data_aug_layer = tf.keras.models.Sequential(data_aug_list)
ds = ds.map(
  lambda x, y: (data_aug_layer(x[None], training=True)[0], y),
  num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
ds = ds.shuffle(
  buffer_size=1000,  # For now a hardcoded value
  reshuffle_each_iteration=True,
).batch(
  32,
  num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
ds = ds.prefetch(
  buffer_size=tf.data.experimental.AUTOTUNE,
)

res = next(iter(ds))  # warning is thrown here

