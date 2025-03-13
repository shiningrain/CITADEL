
import torch
torch.manual_seed(0)
x = torch.ones([10, 13, 3, 3], dtype=torch.bfloat16)
x_trans = torch.transpose(x,2, 3)
x_sum = torch.sum(x_trans, (0, 1, 2))
print(x_sum)

  
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)