
import torch
from earth_mesh.entity.sparse_tensor import SparseTensor 
# SparseTensor is a class that has some custom functions I wrote that operate on sparse tensors

dense_range = torch.arange(100 * 100 * 100).view(100, 100, 100)
sparse_range = dense_range.to_sparse()

indices = torch.randint(0, 100, (100,))

%time torch.index_select(dense_range, 0, indices)
%time torch.index_select(sparse_range, 0, indices)
%time SparseTensor._index_select(SparseTensor(sparse_range), 0, indices)
