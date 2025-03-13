
import torch
import numpy as np
torch.manual_seed(0)

t = torch.load("../masks.pth")
print(torch.unique(t, return_counts=True)[1].cpu().numpy())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)