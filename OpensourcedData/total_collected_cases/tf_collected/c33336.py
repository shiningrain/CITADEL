#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def main():

    a = tf.sparse.SparseTensor(
        indices = [[0, 0, 1], [0, 0, 2], [0, 1, 2], [0, 1, 3], [0, 2, 1], [0, 2, 3]],
        values = [1., 1., 1., 1., 1., 1.],
        dense_shape = [1, 3, 4]
    ); # a.shape = (1,3,4)
    b = tf.constant(np.random.normal(size = (4, 4, 5)), dtype = tf.float32);
    c = tf.linalg.matmul(tf.sparse.to_dense(a),b); # will be succeed
    c = tf.sparse.sparse_dense_matmul(a,b); # will be fail

if __name__ == "__main__":

    main();

