
import torch
in_channels = 4
out_channels = 1
kernel_size = 1
conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
input_tensor = torch.rand([4, 4, 2, 2], dtype=torch.float32)
result = conv2d(input_tensor)
torch.all(result)
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "/home/mist/pytorch/torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch. 
