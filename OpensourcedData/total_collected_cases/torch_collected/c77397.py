
import torch

print(torch.__version__)

a = torch.rand(1, 8192*10, 3) * 100
b = torch.rand(1, 3, 3)

c_1 = torch.einsum('c...i,cji->c...j', a, b)
c_2 = torch.einsum('c...i,cji->c...j', a.cuda(), b.cuda())
c_3 = (a.cuda()[...,None,:] * b.cuda()[:,None,...]).sum(-1)

print((c_1 - c_2.cpu()).abs().max(), (c_1 - c_3.cpu()).abs().max())
