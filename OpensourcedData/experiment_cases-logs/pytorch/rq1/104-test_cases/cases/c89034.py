
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
x = torch.randn(20, dtype=torch.float32, requires_grad=True)
res = torch.sgn(x)
res_bwd = res.grad_fn(torch.ones(res.shape))

# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
r_e_s=res_bwd.detach().func_cls()

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)