
results = dict()
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
arg_class = func_cls(512, 2048, 1)
arg_1=torch.rand([128,512,16,16,16],dtype=torch.float16)
results["time_low"] = arg_class(arg_1)
