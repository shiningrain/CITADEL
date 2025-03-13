
import torch
torch.manual_seed(0)
x = torch.randn(20, dtype=torch.float32, requires_grad=True)
res = torch.sgn(x)
res_bwd = res.grad_fn(torch.ones(res.shape))

# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
print(res_bwd.detach().numpy())

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)