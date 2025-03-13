import tensorflow as tf
import timeit

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((2, )),
    tf.keras.layers.Dense(2048, activation='elu'),
    tf.keras.layers.Dense(2048, activation='elu'),
    tf.keras.layers.Dense(1)
])

opt = tf.keras.optimizers.Adam()


@tf.function
def normal_func(x, y, k):
    print("Tracing normal")
    with tf.GradientTape() as tape:
        loss = tf.constant((0, ), dtype=tf.float32)
        for i in tf.range(k):
            geny = model(tf.expand_dims(x[i], axis=0), training=True)[0]
            loss = loss + tf.square(y[i] - geny)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function(jit_compile=True)
def jit_func(x, y, k):
    print("Tracing jit")
    with tf.GradientTape() as tape:
        loss = tf.constant((0, ), dtype=tf.float32)
        for i in tf.range(k):
            geny = model(tf.expand_dims(x[i], axis=0), training=True)[0]
            loss = loss + tf.square(y[i] - geny)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


x_ = tf.reshape(tf.range(0, 1024, dtype=tf.float32), shape=(512, 2))
y_ = tf.reshape(tf.range(0, 512, dtype=tf.float32), shape=(512, ))

k_ = tf.constant(8)
k2_ = tf.constant(256)
k3_ = tf.constant(512)

# Sanity tests for retracing
normal_func(x_, y_, k_)
normal_func(x_, y_, k2_)
normal_func(x_, y_, k3_)

jit_func(x_, y_, k_)
jit_func(x_, y_, k2_)
jit_func(x_, y_, k3_)

# Actual performance tests
print('Runtime without jit_compile:')
print(timeit.timeit(lambda: normal_func(x_, y_, k_), number=10))
print(timeit.timeit(lambda: normal_func(x_, y_, k2_), number=10))
print(timeit.timeit(lambda: normal_func(x_, y_, k3_), number=10))

print('\nRuntime with jit_compile:')
print(timeit.timeit(lambda: jit_func(x_, y_, k_), number=10))
print(timeit.timeit(lambda: jit_func(x_, y_, k2_), number=10))
print(timeit.timeit(lambda: jit_func(x_, y_, k3_), number=10))

