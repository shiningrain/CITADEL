
import torch

m = torch.nn.Linear(8, 16)
torch.nn.utils.weight_norm(m, dim=1)

x = torch.rand(1, 8).requires_grad_(True)
y = m(x)
y.backward(torch.ones_like(y))
