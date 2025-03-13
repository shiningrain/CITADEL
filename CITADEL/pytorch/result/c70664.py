
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = torch.randn((2,3,128,128), dtype=torch.float16)
func_cls(a, norm='backward')
