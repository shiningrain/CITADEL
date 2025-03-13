
import torch
M = 2121269248
torch.nonzero(torch.ones(M, device='cuda'))
torch.cuda.synchronize()
