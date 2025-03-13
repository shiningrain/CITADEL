
import torch

a = torch.randint(-1, 8, [3, 4, 5], dtype=torch.int8)
b = torch.rand([4, 3, 0], dtype=torch.complex64, requires_grad=True)
torch.tensordot(a, b, dims=[[1, 0], [0, 1]])
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "/Users/distiller/project/pytorch/torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch. 
