
import torch

input = torch.rand([5, 5, 5], dtype=torch.float64, device='cuda', requires_grad=True)
target = torch.rand([5], dtype=torch.float64, device='cuda', requires_grad=True)

res = torch.nn.functional.kl_div(input, target)
print('forward')

res.sum().backward()
# crash
print('backward')
