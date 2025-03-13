import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
a = torch.randn(9, 9)
a = a.cuda()

a.requires_grad = True

b= a+a

func_cls(b)
