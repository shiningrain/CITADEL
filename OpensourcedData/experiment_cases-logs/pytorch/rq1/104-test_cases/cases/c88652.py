
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
np.random.seed(42)
torch.manual_seed(0)

a=func_cls(1,32,10,dtype=int)
b=func_cls(1,32,10,dtype=int)

r_e_s=a==b
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)