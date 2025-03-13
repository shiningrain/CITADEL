
import torch
results = dict()
input_tensor = torch.rand([], dtype=torch.float64)

input = input_tensor.clone()
dim = 0
print(torch.sum(input, dim=dim))
# tensor(0.1512, dtype=torch.float64)
torch.sparse.sum(input_tensor.clone().to_sparse(),dim=dim,)
# RuntimeError: Trying to create tensor with negative dimension -1: [-1, 1]
