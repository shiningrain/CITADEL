import tensorflow as tf
ds = tf.data.Dataset.range(10) # shape=()
ds = ds.batch(2, drop_remainder=True) # shape=(2,)
print(tf.data.experimental.cardinality(ds)) # 5
ds = ds.unbatch() # shape=()
print(tf.data.experimental.cardinality(ds)) # Should be 10, but is -2 (unknown)

