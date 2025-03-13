
import torch

def fn(input):
    fn_res = torch.xlogy(input,2)
    return fn_res

input = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, (input))
