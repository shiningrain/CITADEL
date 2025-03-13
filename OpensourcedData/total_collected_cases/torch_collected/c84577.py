import torch
torch.manual_seed(0)
t = torch.arange(2*3*3*4*4.0).reshape(2, 3, 3, 4, 4)

expected = torch.nn.functional.avg_pool3d(t, (2,2,2))

in_place = torch.zeros([i*2 for i in expected.shape], dtype=expected.dtype)[::2,::2,::2,::2,::2]

result=torch.nn.functional.avg_pool3d(t, (2,2,2), out=in_place)

print(result==expected)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)