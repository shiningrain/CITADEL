
import torch
import os, psutil

a = torch.randn(100, 100, 512)
for _ in range(10):
    torch.linalg.matrix_rank(a)
    print(psutil.Process(os.getpid()).memory_info().rss)
