
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch.nn import CrossEntropyLoss
func_cls(weight=torch.tensor([.2, .3]))(torch.tensor([[1, 2], [3, .4]]), torch.tensor([-100, 1]))
