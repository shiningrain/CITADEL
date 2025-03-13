
import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

device = 'cpu'
input = torch.randn(2, 4, 6, 6, requires_grad=True, device=device)
weight = torch.randn(8, 1, 3, 3, requires_grad=True, device=device)
bias = None
kwargs = {'groups': 4}

output = F.conv2d(input, weight, bias, **kwargs)
grad_output = torch.randn_like(output)

with fwAD.dual_level():
  input_t = torch.randn_like(input)
  weight_t = torch.randn_like(weight)

  input_d = fwAD.make_dual(input, input_t)
  weight_d = fwAD.make_dual(weight, weight_t)

  output = F.conv2d(input_d, weight_d, bias, **kwargs)

  grad_output_d = fwAD.make_dual(grad_output, torch.randn_like(grad_output))

  result = torch.autograd.grad(output, [input, weight], grad_output_d)
  
