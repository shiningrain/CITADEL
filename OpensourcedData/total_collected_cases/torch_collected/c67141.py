
import torch
torch.manual_seed(0)
x = torch.tensor([1+2j])
y = torch.conj(x)
y.add_(2)
print(x.cpu().detach().numpy())


save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)