
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def test():
    arg_1 = torch.rand([2, 3, 5, 5, 0], dtype=torch.float64).clone()
    arg_2 = torch.rand([2, 3, 5, 5], dtype=torch.float64).clone()
    res = func_cls(arg_1,arg_2,)

test()
