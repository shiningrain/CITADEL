
import torch

def test():
    self = torch.rand([2, 1, 4, 5, 4], dtype=torch.float32)
    kernel_size = [2, 2, 1]
    output_size = 2
    random_samples = torch.rand([0, 1, 3], dtype=torch.float32)
    result = torch._C._nn.fractional_max_pool3d(self, kernel_size, output_size, random_samples)

test()
