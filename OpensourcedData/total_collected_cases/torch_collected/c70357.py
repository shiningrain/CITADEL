
import torch
a = torch.sparse_coo_tensor(
  indices=torch.tensor([[1,2,3]]),
  values=torch.tensor([3., 4., 5.], requires_grad=True),
  size=(1000,)
)

a.requires_grad
# True
a.coalesce().values().requires_grad
# True

# When we call without coalescing, gradients are lost
a._values().requires_grad
# False
