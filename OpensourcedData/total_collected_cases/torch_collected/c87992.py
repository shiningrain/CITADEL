
import torch 

generator = torch.Generator(device='cuda').manual_seed(0)
print(torch.rand((8, 77, 1024), device='cuda', generator=generator).abs().sum())
