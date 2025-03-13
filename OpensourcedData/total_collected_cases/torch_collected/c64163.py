
import torch
from torch.autograd import gradcheck
from torch.nn.functional import nll_loss, log_softmax

device = "cuda"
reduction = "mean"

torch.manual_seed(0)
input = log_softmax(torch.rand((1, 2), device=device, dtype=torch.float64), dim=1).requires_grad_(True)
target = torch.randint(0, 2, (1,), device=device, dtype=torch.int64)

gradcheck(lambda input, target: nll_loss(input, target, reduction=reduction), (input, target))
