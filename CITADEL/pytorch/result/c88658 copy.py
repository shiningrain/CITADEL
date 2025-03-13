
from torch import nn
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch
lm_head = func_cls(1536, 250880, bias=False, dtype=torch.bfloat16)
input=torch.ones(size=(8,1024,1536), dtype=torch.bfloat16)
output=lm_head(input)


# crash:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/skyrex05/wangyi/miniconda3/envs/compatibility_test/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/skyrex05/wangyi/miniconda3/envs/compatibility_test/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 114, in forward
#     return F.linear(input, self.weight, self.bias)
# RuntimeError: [enforce fail at alloc_cpu.cpp:75] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 189079224448 bytes. Error code 12 (Cannot allocate memory)


# # FP32 code


# from torch import nn
# import torch
# lm_head = nn.Linear(1536, 250880, bias=False)
# input=torch.ones(size=(8,1024,1536))
# output=lm_head(input)
