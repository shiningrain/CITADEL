
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

in_channels = 105
out_channels = 1
ifm_shape = [1, in_channels, 1, 1]
padding = [8, 5, 9, 4]
groups = 1
kernel_size = [1, 1]
dilation = [1, 1]
stride = [14, 14]

ifm = torch.rand(ifm_shape)
ifm.requires_grad_(True)
padded_ifm = torch.nn.functional.pad(ifm, padding)

op = func_cls(out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups)

res = op(padded_ifm)

grad_in = torch.rand(res.shape)

res.backward(grad_in)
