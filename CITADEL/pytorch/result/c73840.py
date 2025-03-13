
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import sys

# Set value of variables
i = sys.maxsize + 1

self = torch.full((2, 10, 4,), 0.5, dtype=torch.float64, requires_grad=False)
kernel_size = [i]  # Set above
stride = [i]  # Set above
padding = [0]
dilation = [i]  # Set above
ceil_mode = True
func_cls(self, kernel_size, stride, padding, dilation, ceil_mode)
