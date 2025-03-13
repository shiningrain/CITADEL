
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


conv = torch.nn.Conv2d(3, 3, kernel_size=2).cuda()
in_a = torch.randn(4, 3, 64, 64).cuda()
out_a = conv(in_a)
in_b = torch.clone(in_a.permute(0, 2, 3, 1), memory_format=torch.contiguous_format).permute(0, 3, 1, 2)
out_b = conv(in_b)
print(in_a.stride())  # (12288, 4096, 64, 1)
print(in_b.stride())  # (12288, 1, 192, 3)
print(torch.equal(in_a, in_b)) # True
print(torch.equal(out_a, out_b))  # False
