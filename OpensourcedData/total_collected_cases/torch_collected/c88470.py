py
import torch

a = torch.arange(4.0)

not_zero = 0.001

b = torch.where(a != 0, a, not_zero)
c = a.where(a != 0, not_zero)  # Error!

assert b.equal(c)
