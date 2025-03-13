
import torch

t = torch.zeros((1000000, 100), dtype=torch.float32, device=torch.device('cuda:0'))
cpu = torch.device('cpu')

t.to(cpu, non_blocking=True)  # this is correctly to pinned memory

t.to(cpu, dtype=torch.float16, non_blocking=True)  # this will copy the tensor to pageable memory

t.to(dtype=torch.float16).to(cpu, non_blocking=True)  # this will correctly copy the tensor in fp16 to pinned memory.
