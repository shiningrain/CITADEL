
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

tensor = torch.rand(torch.Size([]))
res1 = func_cls(tensor, 0, 0)
# RuntimeError: std::distance(source_dims.begin(), source_iter) == rest_dimINTERNAL ASSERT FAILED at "../aten/src/ATen/native/TensorShape.cpp":2448, please report a bug to PyTorch.
