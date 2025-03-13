
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

def check_correctness(a: torch.Tensor, b:torch.Tensor, expected: int):
    for mkldnn_flag in [True, False]:
        with torch.backends.mkldnn.flags(enabled=mkldnn_flag):
            c = func_cls(a, b)
            assert(torch.all(c == expected))


val = 1024
a = torch.ones(val, val)
b = torch.ones(val, val)

# check_correctness(a, b, expected=val)

a = a.to(torch.bfloat16)
b = b.to(torch.bfloat16)

check_correctness(a, b, expected=val)
