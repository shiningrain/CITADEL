py
import torch

inf = float('inf')
b = torch.tensor([[inf, -inf], [inf, -inf]])
torch.linalg.eigvals(b )
