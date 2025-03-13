
import torch
torch.manual_seed(0)
input = torch.randint(-2,2,[0], dtype=torch.int32)
print(torch.median(input).detach().numpy())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)