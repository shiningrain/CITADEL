
import torch
t = torch.randint(0, 6, size=(10000, 1))
torch.nn.functional.one_hot(t)
