
import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(0)

a=torch.linspace(1,32,10,dtype=int)
b=np.linspace(1,32,10,dtype=int)

print(a==b)
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)