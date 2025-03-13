import torch
from torch import nn


net = nn.Linear(10, 10)#LazyLinear

t = torch.tensor([1.]*10, dtype=torch.double)
net(t)
