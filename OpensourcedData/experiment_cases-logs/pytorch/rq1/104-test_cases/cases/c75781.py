
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

if __name__ == "__main__":

    n = 8
    x = torch.zeros(n).normal_()
    x.requires_grad = True
    z = func_cls(x).sum()
    z.backward()
