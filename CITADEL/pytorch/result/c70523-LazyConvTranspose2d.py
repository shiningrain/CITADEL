
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch import nn

device = "cpu"
print("Inferencedevice:",device)

CH = 64
x=torch.randn(1,CH,4480,2976)
func_cls(out_channels=CH, kernel_size=3, padding=1, bias=False)(x)
print ("Not grouped conv works")
func_cls(out_channels=CH, kernel_size=3, padding=1, groups=CH, bias=False)(x)
print ("This never shows up")
