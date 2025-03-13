
import torch
print(f'Running PyTorch version: {torch.__version__}')

torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

# Dimension of the square sparse matrix
n = 5
# Number of non-zero elements (up to duplicates)
nnz = 8

rowidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
colidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
itemidx = torch.vstack((rowidx,colidx))
xvalues = torch.randn(nnz, dtype=torch.double, device=torchdevice)
x_coo = torch.sparse_coo_tensor(itemidx, xvalues, size=(n,n)).coalesce()
x_csr = x_coo.to_sparse_csr()

t_dense = torch.triu(x_coo.to_dense())+torch.eye(n,device=torchdevice)
t_coo = t_dense.to_sparse_coo()
t_csr = t_dense.to_sparse_csr()
t_bsr = t_csr.to_sparse_bsr(1)

b = torch.randn(n, dtype=torch.double, device=torchdevice).unsqueeze(1)

print('\ntorch.triangular_solve dense:\n===\n',torch.triangular_solve(b,t_dense,upper=True))
print('\ntorch.linalg.solve_triangular dense:\n===\n',torch.linalg.solve_triangular(t_dense,b,upper=True))

try:
  print('\ntorch.triangular_solve coo:\n===\n',torch.triangular_solve(b,t_coo,upper=True))
except Exception as err:
  print('\nException - torch.triangular_solve coo:\n===\n')
  print(Exception, err)
try:
  print('\ntorch.linalg.solve_triangular coo:\n===\n',torch.linalg.solve_triangular(t_coo,b,upper=True))
except Exception as err:
  print('\nException - torch.linalg.solve_triangular coo:\n===\n')
  print(Exception, err)

print('\ntorch.triangular_solve csr:\n===\n',torch.triangular_solve(b,t_csr,upper=True))
try:
  print('\ntorch.linalg.solve_triangular csr:\n===\n',torch.linalg.solve_triangular(t_csr,b,upper=True))
except Exception as err:
  print('\nException - torch.linalg.solve_triangular csr:\n===\n')
  print(Exception, err)

print('\ntorch.triangular_solve bsr:\n===\n',torch.triangular_solve(b.unsqueeze(0),t_bsr,upper=True))
try:
  print('\ntorch.triangular_solve bsr:\n===\n',torch.linalg.solve_triangular(t_bsr,b,upper=True))
except Exception as err:
  print('\nException - torch.triangular_solve bsr:\n===\n')
  print(Exception, err)

try:
  tt_csr= t_csr.detach().clone().requires_grad_(True)
  bb = b.detach().clone().requires_grad_(True)
  test = torch.autograd.gradcheck(torch.triangular_solve, (bb,tt_csr), check_sparse_nnz=True, eps=1e-6, atol=1e-4)
  print('\n=\ncsr torch.triangular_solve',test)
except Exception as err:
  print('\n=\ncsr torch.triangular_solve except')
  print(Exception, err)
