
import numpy as np
import torch

np.random.seed(42)

x = torch.float(torch.from_numpy(np.random.rand(100)))
print(x)

exp_x = torch.exp(x)
print(exp_x)
