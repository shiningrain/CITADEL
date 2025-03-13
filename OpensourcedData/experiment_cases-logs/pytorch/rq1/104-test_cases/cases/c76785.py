
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
input = torch.rand([3, 2], dtype=torch.float64, requires_grad=True)
vec1 = torch.rand([3], dtype=torch.float64, requires_grad=True)
vec2 = torch.rand([2], dtype=torch.complex128, requires_grad=True)

res = func_cls(input, vec1, vec2)
res2 = res.sum()
res2.backward()
# RuntimeError: expected scalar type ComplexDouble but found Double
