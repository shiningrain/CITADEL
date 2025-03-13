
import torch
torch.manual_seed(0)
a = torch.zeros(3, 3, requires_grad=True)
torch.linalg.matrix_norm(a).backward()
print(a.grad)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)