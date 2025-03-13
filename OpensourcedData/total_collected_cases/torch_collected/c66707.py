
import torch

width = 128
input = torch.rand((1,5,width))*0.1
input = input.cuda().half()
normalized_shape = (width,)
weight = torch.ones(width).cuda().half()
bias = torch.zeros(width).cuda().half()
eps = 1e-5
torch.backends.cudnn.enabled = True

output_fp16 = torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
output_fp32 = torch.layer_norm(input.float(), normalized_shape, weight.float(), bias.float(), eps, torch.backends.cudnn.enabled).half()

assert(torch.allclose(output_fp16, output_fp32))
