
In [1]: import torch

In [2]: x = torch.rand(3, 3).t()

In [3]: x.stride()
Out[3]: (1, 3)

In [4]: u = x.triu()

In [5]: u.stride()
Out[5]: (3, 1)

In [6]: l = x.tril()

In [7]: l.stride()
Out[7]: (3, 1)
