import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

# Fails despite being able to work correctly
conv_layer = func_cls(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), dilation=(1, 1), groups=1, bias=True, padding="same")

