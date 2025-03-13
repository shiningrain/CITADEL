
import torch


def function(a, b):
    return torch.cdist(a, b)
    # return torch.cdist(a, b, compute_mode="use_mm_for_euclid_dist")  # works


# input data
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# calculate forward pass
y = torch.nn.Linear(2, 1)(function(a, b))

# calculate gradient
gradient = torch.autograd.grad(
    y,
    a,
    grad_outputs=torch.ones_like(y),
    create_graph=True,
)[0]

# calc loss to reference
grad_ref = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
loss = torch.nn.MSELoss(reduction="mean")(gradient, grad_ref)

loss.backward()  # NotImplementedError: the derivative for '_cdist_backward' is not implemented.
