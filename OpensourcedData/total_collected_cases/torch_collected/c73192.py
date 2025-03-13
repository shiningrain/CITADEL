
import torch

self = torch.full((1, 1, 1, 1, 1,), 1.5e+300, dtype=torch.float64, requires_grad=False)
kernel_size = [536870912, 536870912, 536870912]
stride = [1, 1, 1]
padding = [0, 0, 0]
dilation = [1879048192, 1879048192, 1879048192]
ceil_mode = True
torch._C._nn.max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode)
