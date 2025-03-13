
import torch
m = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
input = torch.randn(1024, 64, 228, 228, device="cuda")
output = m(input)
