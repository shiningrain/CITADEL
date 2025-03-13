import tensorflow as tf

tf.config.run_functions_eagerly(True)


def build_model_fail():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)
    output = tf.keras.layers.Lambda(lambda_fn)(input)
    return tf.keras.Model(inputs=input, outputs=output)


def build_model_success():
    input = tf.keras.Input(dtype=tf.int32, shape=(), batch_size=1)

    # temporarily setting off the eager execution
    # allows the lambda layer to infer the output spec.
    tf.config.run_functions_eagerly(False)
    output = tf.keras.layers.Lambda(lambda_fn)(input)

    # switching back to eager for runtime debugging
    tf.config.run_functions_eagerly(True)

    return tf.keras.Model(inputs=input, outputs=output)


@tf.function
def lambda_fn(input):
    i = tf.constant(0, dtype=tf.int32)
    while i < input:
        tf.print("loop iteration", i)
        i = i + 1
    return input


if __name__ == "__main__":

    # this works
    model = build_model_success()
    model(5)

    # this doesn't work
    model = build_model_fail()
    model(5)


