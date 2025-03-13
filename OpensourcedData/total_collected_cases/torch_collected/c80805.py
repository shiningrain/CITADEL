
import torch

As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)

torch.einsum("bij,bjk->bik", As, Bs)
