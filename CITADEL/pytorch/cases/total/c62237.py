
import torch
import torch.nn.functional as F
torch.manual_seed(0)

t_in = torch.tensor([[[1.0, 2.0]]])
t_out = F.interpolate(t_in, scale_factor=1.00001, recompute_scale_factor=False)

# print(t_in)
print(t_out.cpu().numpy())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)