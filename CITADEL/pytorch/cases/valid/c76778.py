
import torch
input = torch.rand([2], dtype=torch.float64, requires_grad=True)
mat = torch.rand([2, 3], dtype=torch.complex128, requires_grad=True)
vec = torch.rand([3], dtype=torch.complex128, requires_grad=True)

res = torch.addmv(input, mat, vec)
print("addmv SUCCEED!")

res_2 = res.sum()
print("sum SUCCEED!")

res_2.backward()
# addmv SUCCEED!
# sum SUCCEED!
# RuntimeError: Expected isFloatingType(grad.scalar_type()) || (input_is_complex == grad_is_complex) to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
