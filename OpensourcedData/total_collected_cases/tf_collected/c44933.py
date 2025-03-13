    import tensorflow as tf

    print(tf.version.VERSION)  # 2.3.1
    ds = tf.data.Dataset.range(10)  # shape=()
    ds = ds.batch(2)  # shape=(2,)
    print(tf.data.experimental.cardinality(ds))  # 5
    ds = ds.unbatch()  # shape=()
    print(len(list(ds.as_numpy_iterator()))) # 10
    print(tf.data.experimental.cardinality(ds))  # Should be 10, but is -2 (unknown)

