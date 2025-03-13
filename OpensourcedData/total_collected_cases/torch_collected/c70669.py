import torch
a = torch.randn(9, 9)
a = a.cuda()

a.requires_grad = True

b= a+a

torch.logdet(b)
