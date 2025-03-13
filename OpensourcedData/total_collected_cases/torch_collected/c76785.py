
import torch
input = torch.rand([3, 2], dtype=torch.float64, requires_grad=True)
vec1 = torch.rand([3], dtype=torch.float64, requires_grad=True)
vec2 = torch.rand([2], dtype=torch.complex128, requires_grad=True)

res = torch.addr(input, vec1, vec2)
res2 = res.sum()
res2.backward()
# RuntimeError: expected scalar type ComplexDouble but found Double
