    import tensorflow as tf

    input_shape = (100, 100, 3)

    embedding_model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1),
    ])

    input_sequence = tf.keras.layers.Input((None,) + input_shape)
    sequence_embedding = tf.keras.layers.TimeDistributed(embedding_model)
    outputs = sequence_embedding(input_sequence)

    model = tf.keras.Model(inputs=input_sequence, outputs=outputs)

    model.save('model1')
    

    Error:

    
    ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=5. Full shape received: [None, None, 100, 100, 3]
    

2.  Wrapping a pre-trained `tf.keras.applications` model (closer to my actual use case):

    python
    import tensorflow as tf

    input_shape = (224, 224, 3)

    mobilenet = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )

    input_sequence = tf.keras.layers.Input((None,) + input_shape)
    sequence_embedding = tf.keras.layers.TimeDistributed(mobilenet)
    outputs = sequence_embedding(input_sequence)

    model = tf.keras.Model(inputs=input_sequence, outputs=outputs)

    model.save('model2')
    

    Error:

    
    ValueError: Input 0 of layer conv1_pad is incompatible with the layer: expected ndim=4, found ndim=5. Full shape received: [None, None, 224, 224, 3]
    

3.  Saving as an HDF5 file instead:

    python
    import tensorflow as tf

    input_shape = (224, 224, 3)

    mobilenet = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )

    input_sequence = tf.keras.layers.Input((None,) + input_shape)
    sequence_embedding = tf.keras.layers.TimeDistributed(mobilenet)
    outputs = sequence_embedding(input_sequence)

    model = tf.keras.Model(inputs=input_sequence, outputs=outputs)

    model.save('model3.h5', save_format='h5')
    

    This works without errors.

4.  Saving to the SavedModel format works with just dense layers:

    python
    import tensorflow as tf

    input_shape = (100,)

    embedding_model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Dense(units=10)
    ])

    input_sequence = tf.keras.layers.Input((None,) + input_shape)
    sequence_embedding = tf.keras.layers.TimeDistributed(embedding_model)
    outputs = sequence_embedding(input_sequence)

    model = tf.keras.Model(inputs=input_sequence, outputs=outputs)

    model.save('model4')
    
