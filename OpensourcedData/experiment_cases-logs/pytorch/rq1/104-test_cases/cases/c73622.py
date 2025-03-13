
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
A = torch.tensor([[1e-40, 1e-40], [0.0, 0.0]])
B = torch.tile(A, [9,1,1])
x=func_cls(B).cpu().detach().numpy()


D = torch.tile(A, [9,1,1])
D = D.cuda()
y=func_cls(D).cpu().detach().numpy()
print(y)
print(x)
r_e_s=str(y==x)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)