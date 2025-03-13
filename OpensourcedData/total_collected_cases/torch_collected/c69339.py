
import torch
from torch.nn import CrossEntropyLoss
CrossEntropyLoss(weight=torch.tensor([.2, .3]), label_smoothing=0.1)(torch.tensor([[1, 2], [3, .4]]), torch.tensor([-100, 1]))
