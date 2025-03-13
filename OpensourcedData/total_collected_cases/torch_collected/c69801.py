
import torch
torch.manual_seed(0)
ipt = torch.ones((2, 3, 4))
result=torch.nn.Dropout2d(p=0.5)(ipt).cpu().detach().numpy()
print(result)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)