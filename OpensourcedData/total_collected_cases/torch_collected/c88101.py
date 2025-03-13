
import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(0)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
b = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
c = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).cuda()
d = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).cuda()
e = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).detach().numpy()
f = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).detach().numpy()

x = torch.linalg.lstsq(a, b)[0]
y = torch.linalg.lstsq(c, d)[0]
z = np.linalg.lstsq(e, f)[0]

print(x)
print(y)
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)