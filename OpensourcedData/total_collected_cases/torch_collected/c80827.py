
import torch

torch.manual_seed(123)

A = torch.rand(1, 1, 9, 9)

_, I = torch.nn.MaxPool2d(3, 1, return_indices=True)(A)
print("Indices", I)

B = torch.arange(I.numel(), 0, -1).to(torch.float).view(I.shape).detach()
B.requires_grad = True
print("MaxUnPool Input", B)

C = torch.nn.MaxUnpool2d(3, 1)(B, I)
print("MaxUnPool Output", C)

D = C * torch.arange(C.numel()).to(torch.float).view(C.shape)

# now compute the gradient
E = D.sum()
E.backward()

print(B.grad)
save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)