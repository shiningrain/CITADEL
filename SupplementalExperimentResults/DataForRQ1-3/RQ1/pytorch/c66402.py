
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
dtype=torch.complex64
t = torch.tensor([-5.0100e+02-5.0100e+02j, -5.0100e+02+5.0100e+02j,
        -5.0100e+02-1.0012e+03j, -5.0100e+02+1.0012e+03j,
        -5.0100e+02-1.3438e+04j, -5.0100e+02+1.3438e+04j,
        -5.0100e+02-4.9884e+06j, -5.0100e+02+4.9884e+06j,
        -5.0100e+02-1.0000e+20j, -5.0100e+02+1.0000e+20j,
        5.0100e+02-5.0100e+02j,  5.0100e+02+5.0100e+02j,
        5.0100e+02-1.0012e+03j,  5.0100e+02+1.0012e+03j,
        5.0100e+02-1.3438e+04j,  5.0100e+02+1.3438e+04j,])

r_e_s=str(func_cls(t).cpu().detach().numpy())


save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)