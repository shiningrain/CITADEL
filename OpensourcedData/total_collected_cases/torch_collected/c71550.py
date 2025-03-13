
import torch
from torch.nn import functional as F
npos = 1000
logits = torch.zeros((npos, 2), device="cuda:0", requires_grad=True)

for _ in range(10):
    targets = torch.zeros((), dtype=torch.long, device="cuda:0").expand(npos)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
