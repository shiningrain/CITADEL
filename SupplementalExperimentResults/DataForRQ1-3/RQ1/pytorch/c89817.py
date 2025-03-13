
import numpy as np
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch

np.random.seed(42)

x = func_cls(torch.from_numpy(np.random.rand(100)))
print(x)

exp_x = torch.exp(x)
print(exp_x)
