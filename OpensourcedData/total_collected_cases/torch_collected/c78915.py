
import torch
xs = torch.arange(30).to('mps')
print(xs.topk(30))
