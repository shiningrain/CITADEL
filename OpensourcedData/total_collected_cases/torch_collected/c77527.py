
import torch

device = torch.device("cuda")

x = torch.randn(64, 81, 9, 5, device=device)
y = torch.randn(64, 81, 9, 1, device=device)

A = x.transpose(-1, -2) @ x
B = x.transpose(-1, -2) @ y
b = torch.linalg.solve(A, B)
