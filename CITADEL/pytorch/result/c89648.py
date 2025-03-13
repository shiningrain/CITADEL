
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def test():
    self = torch.rand([2, 1, 4, 5, 4], dtype=torch.float32)
    kernel_size = [2, 2, 1]
    output_size = 2
    random_samples = torch.rand([0, 1, 3], dtype=torch.float32)
    result = func_cls(self, kernel_size, output_size, random_samples)

test()
