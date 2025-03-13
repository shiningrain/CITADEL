
import torch

def test():
    arg_1 = torch.rand([2, 3, 5, 5, 0], dtype=torch.float64).clone()
    arg_2 = torch.rand([2, 3, 5, 5], dtype=torch.float64).clone()
    res = torch.linalg.lstsq(arg_1,arg_2,)

test()
