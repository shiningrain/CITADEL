import tensorflow as tf
import numpy as np
x1=tf.stack(list(np.ones([2,0,3]))).shape
tf.compat.v1.disable_eager_execution()
x2=tf.stack(list(np.ones([2,0,3]))).shape
print(x1==x2)

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
