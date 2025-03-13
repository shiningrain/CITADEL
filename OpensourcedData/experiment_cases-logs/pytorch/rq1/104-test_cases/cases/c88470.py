import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

a = torch.arange(4.0)

not_zero = 0.001

b = func_cls(a != 0, a, not_zero)
c = func_cls(a != 0, not_zero)  # Error!

assert b.equal(c)
