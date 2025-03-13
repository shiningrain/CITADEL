import tensorflow as tf

def floordiv(x, y):
    # x // y
    return tf.math.floordiv(x, y)

@tf.function
def floordiv_tffn(x, y):
    # x // y
    return tf.math.floordiv(x, y)

@tf.function(experimental_compile=True)
def floordiv_compiled(x, y):
    # x // y
    return tf.math.floordiv(x, y)

x, y = tf.constant([0., 0.1, 0.9]), 1.
x0=floordiv(x, y)
x1=floordiv_tffn(x, y)

print(x0==x1)

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
