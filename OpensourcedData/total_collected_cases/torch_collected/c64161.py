
import torch
print(torch.__version__)
x = torch.randn(3, 2)
print(x.dtype)
y = torch.where(x < 1e-8, 1e-8, x)  # <-- the error occurs
