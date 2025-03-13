
import torch
from torch.profiler import profile, ProfilerActivity

l = torch.nn.Linear(10, 10).cuda(0)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    l(torch.zeros(10, 10).cuda(0))
