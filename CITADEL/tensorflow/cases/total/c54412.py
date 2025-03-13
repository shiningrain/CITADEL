import tensorflow as tf
tensor = [0,1,2,3]
mask = tf.random.uniform([4], dtype=tf.float64)
x1=tf.boolean_mask(tensor, mask) 
# Outputs: <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)>
print(x1)

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
