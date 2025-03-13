
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
torch.manual_seed(0)
def unique(x):
    return func_cls(x, sorted=False, return_inverse=False, return_counts=True)

s = torch.tensor(0.).cuda()
x = torch.tensor(float('nan')).cuda()

print(unique(s))
a=unique(x) # <- these two calls have different outputs
b=unique(x) # <-
a= np.array([i.cpu().detach().numpy() for i in a])
b= np.array([i.cpu().detach().numpy() for i in b])
r_e_s=str(a==b)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)