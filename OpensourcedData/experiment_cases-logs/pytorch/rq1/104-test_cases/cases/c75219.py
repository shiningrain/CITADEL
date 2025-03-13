
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

device = 'cpu'
input = torch.randn(2, 4, 6, 6, requires_grad=True, device=device)
weight = torch.randn(8, 1, 3, 3, requires_grad=True, device=device)
bias = None
kwargs = {'groups': 4}

output = func_cls(input, weight, bias, **kwargs)
grad_output = torch.randn_like(output)

with fwAD.dual_level():
  input_t = torch.randn_like(input)
  weight_t = torch.randn_like(weight)

  input_d = fwAD.make_dual(input, input_t)
  weight_d = fwAD.make_dual(weight, weight_t)

  output = func_cls(input_d, weight_d, bias, **kwargs)

  grad_output_d = fwAD.make_dual(grad_output, torch.randn_like(grad_output))

  result = torch.autograd.grad(output, [input, weight], grad_output_d)
  
