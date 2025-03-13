
import torch
torch.manual_seed(0)
a = torch.rand([0, 4])
dim = 0
indices = torch.tensor([0, 1])
result=torch.index_select(a, dim, indices)
print(result)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)