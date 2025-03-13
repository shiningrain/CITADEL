
import torch

print(torch.stack([torch.tensor([True], device='cpu'), torch.tensor([True], device='cpu')]))
print(torch.stack([torch.tensor([True], device='mps:0'), torch.tensor([True], device='mps:0')]))
