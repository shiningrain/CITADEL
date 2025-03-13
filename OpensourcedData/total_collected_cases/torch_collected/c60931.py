
# RuntimeError: The size of tensor a (256) must match the size of tensor b (2) at non-singleton dimension 3


# My model looks like this:


import torch
import torch.fft as fft


class PointwiseMultiplication(torch.nn.Module):
    def __init__(self, c, h, w):
        super(PointwiseMultiplication, self).__init__()
        real_matrix = torch.empty(c, h, w)
        imaginary_matrix = torch.empty(c, h, w)
        torch.nn.init.kaiming_normal_(real_matrix)
        torch.nn.init.kaiming_normal_(imaginary_matrix)
        cplx_matrix = torch.complex(real_matrix, imaginary_matrix)
        self.weight = torch.nn.Parameter(cplx_matrix, requires_grad=True)

    def forward(self, x):
        out = fft.fft2(x)
        out = out * self.weight
        return fft.ifft2(out).real
