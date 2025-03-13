
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
input = torch.rand([1, 1, 2, 2], dtype=torch.float32)
indices = torch.randint(-16,1024,[1, 1, 2, 2], dtype=torch.int64)
kernel_size = [16, -1024]
stride = [-16, 1]
result=func_cls(input, indices, kernel_size, stride)
r_e_s=result.shape[-1]

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)