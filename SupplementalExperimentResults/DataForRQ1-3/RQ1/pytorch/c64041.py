
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
torch.manual_seed(0)

t = torch.load("../masks.pth")
r_e_s=func_cls(t, return_counts=True)[1].cpu().numpy()

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)