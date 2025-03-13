
import torch
import functorch
from functorch.compile import memory_efficient_fusion

def fn_drop(x, y):
    return torch.nn.functional.dropout(x) * y

x=torch.ones(4, device="cuda")
y=torch.ones(4,4,device="cuda")
opt_fn=memory_efficient_fusion(fn_drop)
# for _ in range(100):
print(out=opt_fn(x, y))
