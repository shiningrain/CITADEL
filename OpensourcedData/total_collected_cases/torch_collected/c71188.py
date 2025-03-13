
import torch

torch.manual_seed(0)
x = torch.rand((1, 3, 10, 10))
y = torch.nn.functional.interpolate(x, (1,1), align_corners=False, mode='bilinear')
print(y)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)