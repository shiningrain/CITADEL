
import torch
def f(x):
    return torch.ceil(input=x)

# Calling f directly work fine
a = torch.randn(10)
f(a)

# But TorchScipt fails
torch.jit.script(f)
