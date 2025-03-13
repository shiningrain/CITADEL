
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
in_channels = 4
out_channels = 1
kernel_size = 1
conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
input_tensor = torch.rand([4, 4, 2, 2], dtype=torch.float32)
result = conv2d(input_tensor)
func_cls(result)
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "/home/mist/pytorch/torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch. 
