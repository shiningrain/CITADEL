

import torch
a = torch.nn.Parameter(torch.complex(torch.rand(3), torch.rand(3)))
b = torch.tensor(1.0)
c = a * b
c.real = c.real.clamp_(0, 0.1)
c.abs().mean().backward()
