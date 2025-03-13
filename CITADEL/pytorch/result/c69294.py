
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch import fft

def test_function(x, dim=None):
    return func_cls(fft.fftshift(x, dim=dim), dim=dim)

a = torch.rand(4,1,64,64,64).cuda()
torch.testing.assert_close(test_function(a, (2,3,4)), test_function(a, (2,3,4)), check_stride=False)
