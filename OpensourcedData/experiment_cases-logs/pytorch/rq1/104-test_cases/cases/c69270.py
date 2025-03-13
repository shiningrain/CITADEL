
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
tensor = torch.rand(torch.Size([2, 2, 4]), dtype=torch.float32)
sections = 0
res1 = func_cls(tensor, sections)


# import torch
# tensor = torch.rand(torch.Size([2, 2, 4]), dtype=torch.float32)
# sections = 0
# res1 = torch.vsplit(tensor, sections)
