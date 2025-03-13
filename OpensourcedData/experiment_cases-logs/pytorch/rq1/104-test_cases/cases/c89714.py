
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
t = torch.randint(0, 6, size=(10000, 1))
func_cls(t)
