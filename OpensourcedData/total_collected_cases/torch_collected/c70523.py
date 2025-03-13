
import torch
from torch import nn

device = "cpu"
print("Inference device:", device)

CH = 64
x = torch.randn(1, CH, 4480, 2976)
nn.Conv2d(CH, CH, kernel_size=3, padding=1, bias=False)(x)
print ("Not grouped conv works")
nn.Conv2d(CH, CH, kernel_size=3, padding=1, groups=CH, bias=False)(x)
print ("This never shows up")
