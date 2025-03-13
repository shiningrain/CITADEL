
import numpy as np
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from scipy.stats import norm
from torch.fft import rfft, irfft
import torch
device = 'cuda'
pdf = torch.from_numpy(
    norm.pdf(np.linspace(-10, 10, 1000).reshape(-1,1))
).to(device)
n_iter = 50
n_full = pdf.shape[0] * n_iter - n_iter + 1
fft0 = func_cls(pdf, axis=0, n=n_full)
fft1 = func_cls(pdf, axis=0, n=n_full)

assert (torch.allclose(fft0,fft1))


