import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()

np.random.seed(1)
A = np.random.randint(low=0, high=65535, size=100, dtype=np.uint16).reshape(10,10,1)
B = A.copy()
np.allclose(A,B) # is true

# Eager Execution
a_encoded = tf.image.encode_png(A).numpy()
b_encoded = tf.image.encode_png(B).numpy()

print(len(a_encoded),len(b_encoded)) # prints 178 and 278, 278 expected both times
assert(a_encoded == b_encoded) # Fails

# Session Mode
encode_a = tf.image.encode_png(A)
encode_b = tf.image.encode_png(B)

with tf.Session() as sess:
    a_encoded = sess.run(encode_a)
    b_encoded = sess.run(encode_b)

print(len(a_encoded),len(b_encoded)) # prints 178 and 178 but 278 expected 
assert(a_encoded == b_encoded) # True


