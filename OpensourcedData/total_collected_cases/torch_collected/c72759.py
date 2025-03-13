
import torch
from torch.nn.functional import softplus
from matplotlib import pyplot as plt
from torch.autograd import grad

def f1(x):
    return torch.expm1(x).log()

def f2(x):
    return x + (1 - x.neg().exp()).log()

def f3(x):
    big = x > torch.tensor(torch.finfo(x.dtype).max).log()
    return torch.where(
        big,
        f2(x.masked_fill(~big, 1.)),
        f1(x.masked_fill(big, 1.)),
    )

x = torch.linspace(-50, 150, 200, requires_grad=True)
for i, f in enumerate([f1, f2, f3]):
    y = f(softplus(x))
    print(grad(y.sum(), [x])[0].sum())
    plt.plot(x.detach(), y.detach() + i * 2)
plt.show()
