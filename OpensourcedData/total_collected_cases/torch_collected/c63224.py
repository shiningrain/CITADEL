py
import torch
print(torch.__version__)

x = torch.arange(6, dtype=torch.float).reshape(1,3,2,1)
bn = torch.nn.BatchNorm2d(2).eval()
x = x.permute(0,2,1,3)
print(x)
print(bn(x))
