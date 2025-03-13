
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
input = torch.randn(1, 1, 100, 100)
conv = func_cls(out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False)

with torch.no_grad():
    out_ref = conv(input)

conv.to(memory_format=torch.channels_last)
with torch.no_grad():
    out = conv(input)

r_e_s=torch.mean(torch.abs(out - out_ref))


save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)