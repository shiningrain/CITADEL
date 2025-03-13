
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

conv_weight = torch.randn(6, 1, 30, 30)

def model(weights, x):
    conv_weight = weights
    x = F.conv2d(x, conv_weight)
    x = x.view(x.size(0), -1)
    return x

def loss_fun(param, input_tensor):
    out = model(param, input_tensor)
    return out.sin().sum()


input_tensor = torch.rand((1, 1, 32, 32))
vector = torch.ones_like(input_tensor)

conv_weight.requires_grad_()

with fwAD.dual_level():
    dual_input = fwAD.make_dual(input_tensor, vector)
    out = loss_fun(conv_weight, dual_input)
    torch.autograd.grad(out, conv_weight)

"""
# Alternatively: the functorch repro
from functorch import grad, jacfwd, jacrev, jvp, make_functional

def grad_f(input_tensor):
    return grad(loss_fun)(conv_weight, input_tensor)

print(jvp(grad_f, (input_tensor,), (vector,)))
"""
