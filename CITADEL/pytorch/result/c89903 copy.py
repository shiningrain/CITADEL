# Code to reproduce



import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
import torch
np.random.seed(42)


x = np.random.randn(3)
y = np.random.randn(3) + 1j * np.random.randn(3)
g = np.random.randn(3) + 1j * np.random.randn(3)

print("x: ", x)
print("y: ", y)
print("============================")

device = torch.device("cpu:0")
x1 = torch.tensor(x, device=device, requires_grad=True)
y1 = torch.tensor(y, device=device, requires_grad=True)
g1 = torch.tensor(g, device=device)
o1 = func_cls(x1, y1)
o1.backward(g1)
# print("o1: \n", o1.detach().cpu().numpy())
# print("x1g: \n", x1.grad.detach().cpu().numpy())
r_e_s=str(np.isnan(y1.grad.detach().cpu().numpy()))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)