
import torch
torch.manual_seed(0)
import torch.nn as nn
import copy
x = torch.randn(1, 1, 80, 80)

m_fp32 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
m_bf16 = copy.deepcopy(m_fp32).bfloat16()

x_bf16 = x.bfloat16()

y_fp32 = m_fp32(x).sum()
y_fp32.backward()

y_bf16 = m_bf16(x_bf16).sum()
y_bf16.backward()

a=m_fp32.bias.grad.cpu()#.detach().numpy()
b=m_bf16.bias.grad.cpu()#.numpy()
print(a)
print(b)
print(str((a==b).detach().numpy()))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)