import torch
torch.cat([torch.zeros(0, dtype=torch.bool, device="cpu"), torch.zeros(1, dtype=torch.float32, device="cpu")])
torch.cat([torch.zeros(0, dtype=torch.bool, device="cuda:0"), torch.zeros(1, dtype=torch.float32, device="cuda:0")])
