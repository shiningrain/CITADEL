
import torch
torch.manual_seed(0)
A = torch.tensor([[1e-40, 1e-40], [0.0, 0.0]])
B = torch.tile(A, [9,1,1])
x=torch.det(B).cpu().detach().numpy()


D = torch.tile(A, [9,1,1])
D = D.cuda()
y=torch.det(D).cpu().detach().numpy()
print(y)
print(x)
print(str(y==x))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)