
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(torch.sparse_coo_tensor([[]], [], (3,), device='cuda'),0)
