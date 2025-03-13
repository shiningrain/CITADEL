
import torch

x = torch.rand((3, 4)).to_sparse()
y = torch.rand((3, 4))
print(torch.equal(x, y))
