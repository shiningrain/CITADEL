import numpy as np
import tensorflow as tf


LR = 0.01
X = tf.random.normal([100, 10], 0., 1.)
LOG2PI = 1.8378770664093453
WITH_BUG = True
N_ITERS = 100


def gaussian_pdf(samples, mean, covariance):
    n = tf.cast(tf.shape(samples)[1], tf.float32)
    L = tf.linalg.cholesky(covariance)
    alpha = tf.linalg.cholesky_solve(L, tf.transpose(samples - mean))
    data_fit = -0.5 * tf.reduce_sum(tf.transpose(alpha) * (samples - mean), -1)
    regulariser_bug = -0.5 * tf.reduce_sum(tf.linalg.diag_part(tf.math.log(L)), axis=-1)
    regulariser_fine = -0.5 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=-1)
    regulariser = regulariser_bug if WITH_BUG else regulariser_fine
    normaliser = -n * 0.5 * LOG2PI
    return data_fit + regulariser + normaliser


def loss(x, mu, sigma):
    ll = gaussian_pdf(x, mu, sigma)
    return -tf.reduce_mean(ll)


cov_np = np.random.normal(size=(10, 10), scale=0.001) + np.eye(10) * 2.
cov_var = tf.Variable(cov_np, dtype=tf.float32)
mean_var = tf.Variable([3.] * 10, dtype=tf.float32)


def train(x):
    with tf.GradientTape() as t:
        nll = loss(x, mean_var, cov_var)
    dm, dsig = t.gradient(nll, [mean_var, cov_var])
    dm_avg, dcov_avg = tf.reduce_mean(dm).numpy(), tf.reduce_mean(dsig).numpy()
    print(f'Avg grads: mu: {dm_avg}, cov: {dcov_avg}')
    mean_var.assign_sub(LR * dm)
    cov_var.assign_sub(LR * dsig)


for _ in range(N_ITERS):
    print(f'Negative log-likelihood: {loss(X, mean_var, cov_var).numpy()}')
    train(X)

