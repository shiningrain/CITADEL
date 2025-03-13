
import torch

x = torch.randn(3, 5, dtype=torch.float, device='cuda')

b = torch.empty(1, device='cuda')
c = torch.empty(0, device='cuda', dtype=torch.long)

torch.sort(x[:, 0], out=(b, c))

print(b, b.size(), b.stride())
print(c, c.size(), c.stride())

torch.cuda.synchronize()
