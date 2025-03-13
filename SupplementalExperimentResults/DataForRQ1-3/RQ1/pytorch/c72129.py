
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
xx = torch.zeros(3, 4)
xx.foo = 'bar'
func_cls(xx, './_xx.pt')
torch.load('./_xx.pt').foo
