
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = torch.rand(2, 3, 3).to_sparse().requires_grad_(True)
b = torch.rand(2, 3, 3)
c = func_cls(a, b)
loss = c.sum()
loss.backward()
