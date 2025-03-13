
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
input = torch.rand([1, 1])
mat1 = torch.rand([2, 3])
mat2 = torch.rand([3, 3])
res1 = torch.addmm(input, mat1, mat2)
print("addmm pass")
input = input.to_sparse()
mat1 = mat1.to_sparse()
res2 = func_cls(input, mat1, mat2)
