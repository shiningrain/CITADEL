
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

m = func_cls(kernel_size=4, stride=1, padding=0, dilation=1, ceil_mode=True)
input = torch.randn(20, 16, 1)
output = m(input)
