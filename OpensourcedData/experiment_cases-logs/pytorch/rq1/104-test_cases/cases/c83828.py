

import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = torch.nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
b = torch.tensor(1.0)
c = a * b
x=c.real
c.real = func_cls(x,0, 0.1)
c.abs().mean().backward()
