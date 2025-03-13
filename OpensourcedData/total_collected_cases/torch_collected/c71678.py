
import torch
a = torch.rand(2, 3, 3).to_sparse().requires_grad_(True)
b = torch.rand(2, 3, 3)
c = torch.bmm(a, b)
loss = c.sum()
loss.backward()
