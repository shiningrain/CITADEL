
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
input = torch.rand([1, 210120], dtype=torch.float32)
k = 9
dim = 0
func_cls(input.clone(), k, dim=dim)
