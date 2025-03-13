
results = dict()
import torch
import sys
import pickle
sys.path.append('..')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
arg_class = func_cls(out_channels=2048, kernel_size=1)
arg_1=torch.rand([128,512,16],dtype=torch.float16)
results["time_low"] = arg_class(arg_1)
