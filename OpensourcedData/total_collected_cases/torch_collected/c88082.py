
import torch

x = torch.zeros(10)
data_np = x.detach().cpu().numpy()
rev = torch.utils.dlpack.from_dlpack(data_np)
x.data = rev
