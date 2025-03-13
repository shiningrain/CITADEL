
>>> import torch
>>> x=torch.rand(2)
>>> x
tensor([0.8933, 0.5460])
>>> x.to(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream
>>> x.cuda(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: CUDA error: operation would make the legacy stream depend on a capturing blocking stream




![image](https://user-images.githubusercontent.com/21344631/134495818-75a703aa-e87d-4a0d-8d75-a17415c972c1.png)




<!-- If you have a code sample, error messages, stack traces, please provide it here as well -->

## Expected behavior

>>> import torch
>>> x=torch.rand(2)
>>> x
tensor([0.9744, 0.4073])
>>> print(torch.cuda.is_available())
True
>>> x.to(0)
tensor([0.9744, 0.4073], device='cuda:0')
>>> quit();
