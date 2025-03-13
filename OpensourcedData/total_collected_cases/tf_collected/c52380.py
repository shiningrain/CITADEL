import tensorflow as tf


def f(x, y):
    return tf.meshgrid(x, y)


@tf.function
def g(x, y):
    return tf.meshgrid(x, y)


def main():
    print(f"tensorflow version: {tf.version.VERSION}")
    all_values = tf.range(0.0, 1.0, .1)
    x = y = tf.expand_dims(all_values, -1)

    print(f(x, y))  # This works
    print(g(x, y)) # This fails


if __name__ == '__main__':
    main()

