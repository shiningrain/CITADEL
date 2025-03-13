


import torch

def loss(x, w, alpha):
    b = torch.tensor([1 + 1.j], dtype=torch.complex64)
    return (torch.linalg.norm(torch.exp(1.j * torch.tensor([x @ w]) @ alpha) - b)).pow(2)

def main():
    x = torch.nn.Parameter(torch.randn((1,), dtype=torch.float), requires_grad=True)
    alpha = torch.nn.Parameter(torch.tensor([1], dtype=torch.float), requires_grad=True)
    w = torch.tensor([1], requires_grad=False, dtype=torch.float)

    optim = torch.optim.Adam([x], lr=0.1)
    for i in range(1000):
        optim.zero_grad()
        loss_res = loss(x, w, alpha.to(torch.complex64))
        loss_res.backward()
        optim.step()

main()
