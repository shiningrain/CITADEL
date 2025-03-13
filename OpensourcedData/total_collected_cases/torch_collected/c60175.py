import torch

A = torch.randn(2,3,3).cuda()
B = torch.randn(2,3,4).cuda()
X = torch.linalg.solve(A, B)
