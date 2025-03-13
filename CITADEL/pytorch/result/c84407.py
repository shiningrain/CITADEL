import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)

device = torch.device('cuda')
t = torch.tensor([i/50 for i in range(50)])
input = torch.sin(t)
input = torch.stack([input for i in range(64*512)], dim=0)
input = input.to(device)
import time
start = time.time()
stft = func_cls(input, 16, 10)
a1=time.time()-start # 0.00044
func_cls(stft, 16, 10) - input
b1=time.time()-start # 0.5130

device = torch.device('cpu')
t = torch.tensor([i/50 for i in range(50)])
input = torch.sin(t)
input = torch.stack([input for i in range(64*512)], dim=0)
input = input.to(device)
import time
start = time.time()
stft = func_cls(input, 16, 10)
a2=time.time()-start # 0.0399
func_cls(stft, 16, 10) - input
b2=time.time()-start # 0.1512
r_e_s=b1-b2

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
