
import torch
a = torch.rand([0, 3])
torch.amax(a)
# tensor(-1.8891e+26)
# print(torch.amin(a))
# # tensor(9.1477e-41)


# import torch
# a = torch.rand([0, 3])
# print(torch.max(a))
# RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
