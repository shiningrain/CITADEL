


import tensorflow as tf
with tf.device('/CPU'):
    arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    x1 = tf.image.convert_image_dtype(arg_0, dtype=tf.uint32, saturate=-1).numpy()




import tensorflow as tf
with tf.device('/GPU:0'):
    arg_0 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    x2 = tf.image.convert_image_dtype(arg_0, dtype=tf.uint32, saturate=-1).numpy()
print(x1-x2)


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
