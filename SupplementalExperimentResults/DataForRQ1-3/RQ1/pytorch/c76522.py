
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
a = torch.tensor([-2.,-1.,0.,1.,2.], requires_grad=True)
b = torch.full((5,), -1.)
b.requires_grad = True
c = torch.full((5,), 1.)
c.requires_grad = True
x1=a.grad
y1=b.grad
z1=c.grad
func_cls(a, b,c).sum().backward()
a.grad, b.grad, c.grad = None, None, None
torch.min(torch.max(a, b), c).sum().backward()
x2=a.grad
y2=b.grad
z2=c.grad
r_e_s=str(x1==x2)#,str(y1==y2),str(z1==z2)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)