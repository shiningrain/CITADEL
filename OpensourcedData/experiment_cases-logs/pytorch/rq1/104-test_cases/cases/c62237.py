
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch.nn.functional as F
torch.manual_seed(0)

t_in = torch.tensor([[[1.0, 2.0]]])
t_out = func_cls(t_in, scale_factor=1.00001, recompute_scale_factor=False)

# print(t_in)
r_e_s=t_out.cpu().numpy()

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)