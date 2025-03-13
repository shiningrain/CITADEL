
import torch
# nightly 1.10

# fix is next line
# torch.cuda.set_device("cuda:1")
device = torch.device("cuda:1")

a = torch.randn(20, 50)
b = a @ a.t() + 1e-3 * torch.eye(20)
b = b.to(device)
torch.linalg.eigh(b)
