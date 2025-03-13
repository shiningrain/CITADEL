
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

x = torch.rand((3, 4)).to_sparse()
y = torch.rand((3, 4))
print(func_cls(x, y))
