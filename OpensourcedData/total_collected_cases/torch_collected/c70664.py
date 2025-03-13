
import torch
a = torch.randn((2,3,128,128), dtype=torch.float16)
torch.fft.rfft2(a, norm='backward')
