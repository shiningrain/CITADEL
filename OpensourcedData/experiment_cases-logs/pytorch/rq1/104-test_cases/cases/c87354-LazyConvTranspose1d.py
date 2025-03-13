
# import torch
# import numpy as np
# x = torch.rand(1, 32, 512, 512, 256).to('cuda:0')
# m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0,stride=1,bias=False).to('cuda:0')
# x = m(x)  # Assert!!
# print(x)
# numpy_x = x.cpu().detach().numpy()
# print(np.where(numpy_x != 0), numpy_x.shape)


import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
x = torch.rand(1, 32, 512, 512, 256)
m = func_cls(out_channels=1, kernel_size=1, padding=0, stride=1, bias=False)
x = m(x)  # Assert!!
print(x)
numpy_x = x.cpu().detach().numpy()
r_e_s=np.where(numpy_x != 0)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
