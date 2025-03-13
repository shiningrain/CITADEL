
import torch
input = torch.rand([1, 210120], dtype=torch.float32)
k = 9
dim = 0
torch.kthvalue(input.clone(), k, dim=dim)
