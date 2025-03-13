
import torch
# device = torch.device('cuda')  # This will work fine
device = torch.device('cpu')  # This will give an error
x = torch.arange(10, device=device)
idx1 = torch.tensor([4, 2, 0, 8, 0, 1], device=device)
idx2 = torch.tensor([4, 2, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], device=device)  # Too long for CPU

print('idx1.numel() =', idx1.numel())
print(x.scatter_(0, idx1, 0))
# idx1.numel() = 6
# tensor([0, 0, 0, 3, 0, 5, 6, 7, 0, 9], device='cuda:0')

print('idx2.numel() =', idx2.numel())
print(x.scatter_(0, idx2, 0))
# idx2.numel() = 14
# tensor([0, 0, 0, 3, 0, 5, 6, 7, 0, 9], device='cuda:0')
