
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

sparse_tensor = torch.sparse_coo_tensor([[1,2]], [1,5], (3,))
func_cls(sparse_tensor, -1, 1)
