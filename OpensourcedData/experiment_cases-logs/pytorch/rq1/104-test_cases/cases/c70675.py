
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
a = torch.rand([0, 4])
dim = 0
indices = torch.tensor([0, 1])
result=func_cls(a, dim, indices)
r_e_s=str(type(result))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)