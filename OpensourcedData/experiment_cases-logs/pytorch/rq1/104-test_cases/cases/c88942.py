
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def test():
    arg_1 = torch.rand([5, 5], dtype=torch.float64).to_sparse()
    res = func_cls(arg_1,False,)

test()
