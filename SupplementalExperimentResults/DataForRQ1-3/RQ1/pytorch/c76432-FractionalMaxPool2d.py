
import numpy as np
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch
np.random.seed(10)
# shape = (1, 5, 5, 3)
# arr = np.random.randn(*shape).transpose(0, 3, 1, 2) * 10                #（1）

shape = (1, 3, 5, 5)
arr = np.random.randn(*shape) * 10                                          #（2）

kernel_size = 3
padding = 0
dilation = (2, 2)
stride = 1
ceil_mode = False
cpu_x = torch.tensor(arr, dtype=torch.float64, device="cpu")
cuda_x = torch.tensor(arr, dtype=torch.float64, device="cuda")
assert np.allclose(cpu_x.detach().cpu().numpy(), cuda_x.detach().cpu().numpy(), 1e-4, 1e-4)
m = func_cls(kernel_size)
cpu_y = m(cpu_x)
cuda_y = m(cuda_x)
print("cpu y = \n", cpu_y)
print("cuda y = \n", cuda_y)
result=cpu_y.detach().cpu().numpy()- cuda_y.detach().cpu().numpy()
r_e_s=result


save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)