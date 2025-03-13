py
import torch

input = torch.FloatTensor([1, 3])
torch.unique(input, sorted=False)
# tensor([3., 1.])
torch.unique(input.cuda(), sorted=False)
# tensor([1., 3.], device='cuda:0')
