
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(input=torch.ones((1,1,1)),weight=torch.ones((1,1,1)),groups=0)
