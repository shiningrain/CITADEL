
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])

cpu = func_cls(2, stride=2)
cuda = func_cls(2, stride=2).to("cuda:0")

output = torch.rand(torch.Size([1, 1, 2, 2]), dtype=torch.float32)
indices = torch.randint(-32768,32768,torch.Size([1, 1, 2, 2]), dtype=torch.int64)

cuda_res = cuda(output.cuda(), indices.cuda())
print(cuda_res)

cpu_res = cpu(output, indices)
print(cpu_res)
