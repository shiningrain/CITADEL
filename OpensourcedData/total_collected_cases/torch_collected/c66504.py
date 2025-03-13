
import os
import torch
import torch.nn
import torch.distributed

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "45678"
world_size = 1
rank = 0

torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)
device = 'cuda'

net = torch.nn.BatchNorm1d(128) # (128)
# net = torch.nn.LazyBatchNorm1d()
net.to(device=device)
net = torch.nn.parallel.DistributedDataParallel(
    net,
    device_ids=[rank],
    output_device=rank,
)
net.train()

input1 = torch.rand(64, 128, 1, device=device) # torch.rand(64, 128, device=device)

o = net(input1).sum() + net(input1).sum()
o.backward()
