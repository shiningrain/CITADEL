
import torch
y = torch.ones(2,1, device='mps')
y_pad = torch.nn.functional.pad(y, (1, 1, 1, 1), 'constant', 0)
print(y_pad)
