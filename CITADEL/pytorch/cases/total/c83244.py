
import torch
results={}
arg_1 = [1]
arg_2 = "nearest"
arg_class = torch.nn.Upsample(scale_factor=arg_1,mode=arg_2,)
arg_3 = torch.rand([1, 1, 2, 2], dtype=torch.float32)
results['res'] = arg_class(*arg_3)
#TypeError: float() argument must be a string or a number, not 'list'
