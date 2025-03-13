

import torch
torch.manual_seed(0)
t=torch.ones((3,3), dtype=torch.float64)
a=t.exp().cpu().numpy()
print(a)

t=torch.ones((3,3), dtype=torch.float32)
b=t.exp().cpu().numpy()

print(b)

value=a-b
print(value)


