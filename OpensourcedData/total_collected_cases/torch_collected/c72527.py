
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


if __name__ == '__main__':
    conv = nn.Conv2d(3, 3, kernel_size=1)

    x = torch.ones(4, 3, 224, 224, requires_grad=True)
    x = checkpoint(conv, x)

    loss = torch.mean(x)
    loss.backward()
