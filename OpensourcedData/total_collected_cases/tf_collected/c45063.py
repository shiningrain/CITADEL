import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

data, data_info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
data = data.map(lambda x, _: tf.cast(x, tf.float32) / 255.)
data_shape = data_info.features["image"].shape
dimension = tf.reduce_prod(data_shape).numpy()

latent_distribution = tfp.distributions.MultivariateNormalDiag(
    loc=[0.] * dimension,
    scale_diag=[1.] * dimension,
)

input = tf.keras.Input(shape=data_shape)
state = input
state = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, use_bias=False)(state)
state = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)(state)
state = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)(state)
state = tf.keras.layers.Activation(tf.nn.softplus)(state)  # Essential to trigger bug
state = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, use_bias=False)(state)
state = tf.keras.layers.Flatten()(state)
f = tf.keras.Model(inputs=input, outputs=state)

optimizer = tf.keras.optimizers.Adam()

@tf.function  # Essential to trigger bug
def train_step(data):
    with tf.GradientTape() as tape:
        tape.watch(f.trainable_variables)

        with tf.GradientTape() as c_tape:
            c_tape.watch(data)

            with tf.GradientTape() as a_tape:
                a_tape.watch(data)
                b = f(data)
            a = a_tape.gradient(b, data)
            a_flat = tf.reshape(a, (-1, dimension))

        c = c_tape.batch_jacobian(a, data)
        c = tf.reshape(c, (-1, dimension, dimension))

        d = latent_distribution.log_prob(a_flat)
        _, e = tf.linalg.slogdet(c)
        ff = tf.reduce_mean(d + e)

        loss = -ff

    gradients = tape.gradient(loss, f.trainable_variables)
    optimizer.apply_gradients(zip(gradients, f.trainable_variables))

train_data = data.batch(1)
for batch in train_data:
    train_step(batch)

