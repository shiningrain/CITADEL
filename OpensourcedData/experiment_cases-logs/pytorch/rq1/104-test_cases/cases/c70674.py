
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = torch.rand([0, 3])
print(func_cls(a))
# tensor(-1.8891e+26)
print(torch.amin(a))
# tensor(9.1477e-41)


import torch
a = torch.rand([0, 3])
print(func_cls(a))
# RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
