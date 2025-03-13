import tensorflow as tf
nbins = -16
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=nbins)
x1=indices.numpy()
print(x1)

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
