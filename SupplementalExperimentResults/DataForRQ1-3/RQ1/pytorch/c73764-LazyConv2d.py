
import torch as ch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

num_convs = 200
in_channels = 512
out_channels = 512
device = 'cuda:0'
x = ch.randn(1, in_channels, 32, 32).to(device)

convs = [func_cls(out_channels=out_channels, kernel_size=3, bias=False).to(device) for _ in range(num_convs)]

fused_conv = func_cls(out_channels=convs[0].out_channels * num_convs, groups=convs[0].groups * num_convs, kernel_size=convs[0].kernel_size, bias=convs[0].bias).to(device)

def run_in_streams(convs, x):
    out = []
    for ind, stream in enumerate([ch.cuda.Stream(device) for _ in range(len(convs))]):
        with ch.cuda.stream(stream):
            out.append(convs[ind](x))
    ch.cuda.synchronize()
    return ch.stack(out)

for _ in range(2):  # warm-up CUDA
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            out_fused = fused_conv(ch.cat([x] * num_convs, dim=1))
print('Use a single convolution with groups')