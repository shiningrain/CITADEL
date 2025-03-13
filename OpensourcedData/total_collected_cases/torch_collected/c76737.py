
import torch
from torch.utils import checkpoint

c = torch.nn.Linear(10,10).cuda()
x = torch.rand(10, 10, requires_grad=True).cuda()

checkpoint.checkpoint(c.forward, x, use_reentrant=False, preserve_rng_state=False)
