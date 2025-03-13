
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
t = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1, 2], dtype=torch.int16)
r_e_s=str(func_cls(t).dtype)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)