
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
M = 2121269248
func_cls(torch.ones(M, device='cuda'))
torch.cuda.synchronize()
