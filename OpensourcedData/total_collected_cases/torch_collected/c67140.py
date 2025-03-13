
import numpy as np
from scipy.stats import norm
from torch.fft import rfft, irfft
import torch
device = 'cuda'
pdf = torch.from_numpy(
    norm.pdf(np.linspace(-10, 10, 1000).reshape(-1,1))
).to(device)
n_iter = 50
n_full = pdf.shape[0] * n_iter - n_iter + 1
fft0 = rfft(pdf, axis=0, n=n_full)
fft1 = rfft(pdf, axis=0, n=n_full)

assert (torch.allclose(fft0,fft1))


