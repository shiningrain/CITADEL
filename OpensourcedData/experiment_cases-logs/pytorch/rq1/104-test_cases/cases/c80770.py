
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def fn(input):
    fn_res = func_cls(input,2)
    return fn_res

input = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input))
