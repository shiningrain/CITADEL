import tensorflow as tf

train_images = tf.keras.preprocessing.image_dataset_from_directory(
    'images',
    labels=None,
)

