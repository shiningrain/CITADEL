
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
results={}
arg_1 = [1]
arg_2 = "nearest"
arg_class = func_cls(arg_1)
arg_3 = torch.rand([1, 1, 2, 2], dtype=torch.float32)
results['res'] = arg_class(*arg_3)
#TypeError: float() argument must be a string or a number, not 'list'
