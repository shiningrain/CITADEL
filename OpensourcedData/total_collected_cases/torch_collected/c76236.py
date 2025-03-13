
import torch
print(torch.nonzero(torch.tensor(0)))
# output tensor([], size=(0, 0), dtype=torch.int64)
print(torch.nonzero(torch.tensor(1)))
# output tensor([], size=(1, 0), dtype=torch.int64)
