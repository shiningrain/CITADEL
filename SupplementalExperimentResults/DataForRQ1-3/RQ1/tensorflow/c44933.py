import tensorflow as tf
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

print(tf.version.VERSION)  # 2.3.1
ds = tf.data.Dataset.range(10)  # shape=()
ds = ds.batch(2)  # shape=(2,)
print(func_cls(ds))  # 5
ds = ds.unbatch()  # shape=()
print(len(list(ds.as_numpy_iterator()))) # 10
r_e_s=func_cls(ds)  

import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
import pickle
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)