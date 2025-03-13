
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import time
device = torch.device('cuda')
length = 10000
hidden = 6
batch = 256
signal = torch.tensor([[[i for i in range(length)] for dim in range(hidden)] for b in range(batch)], device=device).float()
time_ls = []
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    spec = torch.fft.rfft(signal)
    # torch.fft.irfft(spec)
    torch.cuda.synchronize()
    end  = time.time()
    time_ls.append(end - start)
time_ls = torch.tensor(time_ls)
time_ls = time_ls.sort()[0]
time_ls = time_ls[5:-5]
print(time_ls.mean(), time_ls.std())

time_ls = []
signal = signal.reshape(-1, length)
n_fft = 256
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    spec = func_cls(signal, n_fft=n_fft, hop_length=n_fft, center=False, onesided=True, return_complex=None)
    # torch.istft(spec, n_fft=n_fft, hop_length=n_fft, center=False, onesided=True)
    torch.cuda.synchronize()
    end  = time.time()
    time_ls.append(end - start)
time_ls = torch.tensor(time_ls)
time_ls = time_ls.sort()[0]
time_ls = time_ls[5:-5]
r_e_s=time_ls.mean(), time_ls.std()
