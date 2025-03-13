
import torch

sparse_tensor = torch.sparse_coo_tensor([[1,2]], [1,5], (3,))
torch.clamp(sparse_tensor, -1, 1)
