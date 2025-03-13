
import torch

if __name__ == "__main__":

    n = 8
    x = torch.zeros(n).normal_()
    x.requires_grad = True
    z = torch.fft.irfft(x).sum()
    z.backward()
