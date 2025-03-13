py
>>> import torch
>>> c = torch.nn.Conv2d(3, 3, 256)
>>> x = torch.rand(1, 3, 512, 512)
>>> 
>>> c(x)
