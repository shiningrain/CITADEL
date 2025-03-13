
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
input = torch.tensor([[1.0]*2, [3]*2], requires_grad=True).cuda() 
output = func_cls(torch.matmul(input, input))
print(output)
