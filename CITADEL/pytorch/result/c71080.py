
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
func_cls(inputs, filters, padding=1, groups=0)
# torch.nn.functional.conv2d(inputs, filters, padding=1, groups=0)
# floating point exception
