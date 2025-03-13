import tensorflow as tf
import os
import psutil
process = psutil.Process(os.getpid())

for i in range(int(1e7)):
    with tf.name_scope("first"):
        with tf.name_scope("second"):
            pass

print(process.memory_info().rss // 1000000)

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)