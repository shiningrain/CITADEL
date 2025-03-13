
import torch
A = torch.rand(10, 2, requires_grad=True)
B = torch.rand(10, 1)
torch.linalg.lstsq(A, B)
