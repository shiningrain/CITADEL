import tensorflow as tf


def main():
    variable = tf.Variable(1.0)
    inputs = (
        tf.constant(tf.random.uniform((0, 4))),
        tf.constant(tf.random.uniform((0, 3))),
    )

    with tf.GradientTape(persistent=True) as tape:
        outputs = variable * tf.pow(tf.concat(inputs, axis=-1), 2.0)

    jacobians_1 = tape.jacobian(
        outputs,
        variable,
        experimental_use_pfor=True,
    )
    print(jacobians_1)
    print("tape.jacobians(..., experimental_use_pfor=True) works!")

    try:
        jacobians_2 = tape.jacobian(
            outputs,
            variable,
            experimental_use_pfor=False,
        )
        print(jacobians_2)
        print("tape.jacobians(..., experimental_use_pfor=False) works!")
    except TypeError:
        print("tape.jacobians(..., experimental_use_pfor=False) doesn't work!")
        raise


if __name__ == '__main__':
    main()

