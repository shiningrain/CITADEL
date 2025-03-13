
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
input_tensor = torch.randint(-1,1,[3], dtype=torch.int64)
input = input_tensor.clone()
r = 100
print(func_cls(input, r=r))
# killed
