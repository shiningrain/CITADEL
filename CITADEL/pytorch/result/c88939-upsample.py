
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def test():
    arg_1 = torch.rand([1, 10, 540, 540], dtype=torch.bfloat16).clone()
    res = func_cls(arg_1, 2, 'bilinear', True)

test()
