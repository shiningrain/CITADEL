
import torch
input = torch.rand([1, 1])
mat1 = torch.rand([2, 3])
mat2 = torch.rand([3, 3])
res1 = torch.addmm(input, mat1, mat2)
print("addmm pass")
input = input.to_sparse()
mat1 = mat1.to_sparse()
res2 = torch.sspaddmm(input, mat1, mat2)
