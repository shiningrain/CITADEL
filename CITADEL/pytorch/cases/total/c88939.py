
import torch

def test():
    arg_1 = torch.rand([1, 10, 540, 540], dtype=torch.bfloat16).clone()
    res = torch.nn.functional.interpolate(arg_1,2,mode='bilinear',align_corners=True)

test()
