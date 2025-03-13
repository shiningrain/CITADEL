
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
x = torch.ones([10, 13, 3, 3], dtype=torch.bfloat16)
x_trans = func_cls(x,2, 3)
x_sum = torch.sum(x_trans, (0, 1, 2))
r_e_s=x_sum

  
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)