
import tensorflow as tf
image = tf.constant([[[254 + 2j]], [[83]], [[72]]], dtype=tf.complex64)
dtype = tf.float64
out = tf.image.convert_image_dtype(image, dtype)
print(out)


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
