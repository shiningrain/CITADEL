import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch import nn


net = func_cls(out_features=10)#LazyLinear

t = torch.tensor([1.]*10, dtype=torch.double)
net(t)
