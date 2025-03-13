
$ 
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> a = torch.Tensor([[1,2],[3,4]])
>>> b = torch.Tensor([[5],[6]])
>>> torch._weight_norm(a,b,dim=1)
tensor([[1.5811, 2.2361],
        [5.6921, 5.3666]])



#### torch 1.12

$ 
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> a = torch.Tensor([[1,2],[3,4]])
>>> b = torch.Tensor([[5],[6]])
>>> torch._weight_norm(a,b,dim=1)
tensor([[1.5811, 2.6833],
        [4.7434, 5.3666]])
