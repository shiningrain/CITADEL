
import torch

def test():
    arg_1 = torch.rand([5, 5], dtype=torch.float64).to_sparse()
    res = torch.svd_lowrank(arg_1,False,)

test()
