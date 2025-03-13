
import torch

tensor = torch.rand(torch.Size([]))
res1 = torch.movedim(tensor, 0, 0)
# RuntimeError: std::distance(source_dims.begin(), source_iter) == rest_dimINTERNAL ASSERT FAILED at "../aten/src/ATen/native/TensorShape.cpp":2448, please report a bug to PyTorch.
