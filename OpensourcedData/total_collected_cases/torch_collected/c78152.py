
import torch

input = torch.rand([0], dtype=torch.float32).requires_grad_()
res = torch.nn.functional.gelu(input, )

res.sum().backward()
# Segmentation fault (core dumped)
