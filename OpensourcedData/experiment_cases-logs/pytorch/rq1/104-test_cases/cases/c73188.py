
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

input = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
num_groups = 0
weight = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
bias = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
eps = 0
cudnn_enabled = False
func_cls(input, num_groups, weight, bias, eps, cudnn_enabled)
