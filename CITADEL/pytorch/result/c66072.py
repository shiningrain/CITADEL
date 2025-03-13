
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
print(torch.__version__) # 1.9.1+cu102
torch.manual_seed(0)

A = torch.randn(11, 100, 100, device='cuda')
A = A @ A.transpose(-2, -1)
b = torch.randn(11, 100, 200, device='cuda')

assert (torch.allclose(func_cls(b, A), func_cls(b.cpu(), A.cpu()).cuda())) # False, but should be True

assert (torch.allclose(func_cls(b[0], A[0]), func_cls(b[0].cpu(), A[0].cpu()).cuda())) # True
