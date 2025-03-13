
import torch

m = torch.nn.MaxPool1d(kernel_size=4, stride=1, padding=0, dilation=1, ceil_mode=True)
input = torch.randn(20, 16, 1)
output = m(input)
