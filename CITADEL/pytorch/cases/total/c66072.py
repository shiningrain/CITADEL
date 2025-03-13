
import torch
print(torch.__version__) # 1.9.1+cu102
torch.manual_seed(0)

A = torch.randn(11, 100, 100, device='cuda')
A = A @ A.transpose(-2, -1)
b = torch.randn(11, 100, 200, device='cuda')

assert (torch.allclose(torch.cholesky_solve(b, A), torch.cholesky_solve(b.cpu(), A.cpu()).cuda())) # False, but should be True

assert (torch.allclose(torch.cholesky_solve(b[0], A[0]), torch.cholesky_solve(b[0].cpu(), A[0].cpu()).cuda())) # True
