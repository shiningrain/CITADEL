
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
def f(x):
    return torch.ceil(input=x)

# Calling f directly work fine
a = torch.randn(10)
f(a)

# But TorchScipt fails
func_cls(f)
