
import torch
import time

elapsed = 0.0
runs = 100
for i in range(runs):
    tensor = torch.rand(100_000_000, device='cuda:0')

    torch.cuda.synchronize()
    start = time.time()

    sorted, _ = tensor.sort()
    second_smallest_el = sorted[1]

    torch.cuda.synchronize()
    end = time.time()
    elapsed += end - start

print(elapsed/runs)  # 0.03 sec

2) Second approach via `torch.kthvalue`:

import torch
import time

elapsed = 0.0
runs = 100
for i in range(runs):
    tensor = torch.rand(100_000_000, device='cuda:0')

    torch.cuda.synchronize()
    start = time.time()

    val, _ = torch.kthvalue(tensor, 2)

    torch.cuda.synchronize()
    end = time.time()
    elapsed += end - start

print(elapsed/runs)  # 0.68 sec
