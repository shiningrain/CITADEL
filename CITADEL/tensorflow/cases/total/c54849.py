import tensorflow as tf
import numpy as np
splits = [-16, 4, 2, 5, 5, 7]
result = tf.ragged.row_splits_to_segment_ids(splits) # pass, but it should throw ValueError as splits starts with -16
print(result) 


import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
