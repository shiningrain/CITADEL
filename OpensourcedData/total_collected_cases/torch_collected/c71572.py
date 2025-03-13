
import torch
x = torch.rand(10).to('cuda')
y = torch.tensor([11]).to('cuda')
print(x[y])
