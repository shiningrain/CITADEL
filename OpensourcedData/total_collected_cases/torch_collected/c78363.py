
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import torch
import numpy as np

mat1 = np.diag([1, 0.5, 0.25, 0.125])
mat2 = np.diag([1,2,3,4])

for device in ['cpu', 'mps']:
    print('\n', device)
    mats = torch.tensor(np.array([mat1, mat2]), dtype=torch.float, device=device)
    correct = np.linalg.inv(mats.cpu().numpy())
    for i in range(2):
        print('error for matrix', i, ' is ', (torch.linalg.inv(mats[i]).cpu().numpy() - correct[i]))
