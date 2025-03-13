
import torch
print(torch.__version__)

torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
import torch_sparse

# Convenience wrapper around torch_sparse index_select
def ts_index_select(A,sdim,idx):
  Ats = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(A)
  Ats_select = torch_sparse.index_select(Ats,sdim,idx)
  row, col, value = Ats_select.coo()
  As_select = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), value, (Ats_select.size(0), Ats_select.size(1)))
  return As_select

# Dimension of the square sparse matrix
n = 1000000
# Number of non-zero elements (up to duplicates)
nnz = 100000
# Number of selected indices (up to duplicates)
m = 10000

rowidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
colidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
itemidx = torch.vstack((rowidx,colidx))
xvalues = torch.randn(nnz, device=torchdevice)
SparseX = torch.sparse_coo_tensor(itemidx, xvalues, size=(n,n)).coalesce()
print('SparseX:',SparseX)

selectrowidx = torch.unique(torch.randint(low=0, high=n, size=(m,), device=torchdevice), sorted=True)

print('\nRunning index_select from PyTorch')
%timeit SparseXsub1 = SparseX.index_select(0,selectrowidx)

print('\nRunning index_select from torch_sparse')
%timeit SparseXsub2 = ts_index_select(SparseX,0,selectrowidx)
