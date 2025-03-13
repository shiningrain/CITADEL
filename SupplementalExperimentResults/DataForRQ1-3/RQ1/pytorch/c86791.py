
from torch import nn
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

print(torch.__version__)

with torch.no_grad():
    x = torch.zeros(1, 3, 640, 640)
    model = func_cls(3, 3, kernel_size=1)
    script_model = torch.jit.trace(model, x)
    optimized_traced = optimize_for_mobile(script_model, backend='vulkan')
    optimized_traced._save_for_lite_interpreter("sample_data/t.ptl")
