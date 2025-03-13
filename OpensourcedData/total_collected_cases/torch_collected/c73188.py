
import torch

input = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
num_groups = 0
weight = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
bias = torch.full((4, 6, 5,), 1, dtype=torch.float32, requires_grad=False)
eps = 0
cudnn_enabled = False
torch.group_norm(input, num_groups, weight, bias, eps, cudnn_enabled)
