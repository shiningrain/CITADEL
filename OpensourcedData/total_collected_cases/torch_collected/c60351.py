
import torch

N = 100  # this value is arbitrary
print('Expected value :', torch.sum(torch.arange(N).float() * torch.arange(N).float()))  # naive implementation of dot product
print('torch.int      :', torch.dot(torch.arange(N), torch.arange(N)))
print('torch.float64  :', torch.dot(torch.arange(N).double(), torch.arange(N).double()))
print('torch.float32  :', torch.dot(torch.arange(N).float(), torch.arange(N).float()))
