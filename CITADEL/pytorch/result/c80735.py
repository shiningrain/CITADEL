
import time
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import torch as th
import torch

N = 100
X = th.randn(153531, 4, 4, device="cuda")

func_cls(X)

th.cuda.synchronize()
start = time.time()

for _ in range(N):
    func_cls(X)

th.cuda.synchronize()
end = time.time()

print("Time per inverse (ms):", 1000 * (end - start) / N)
print("PyTorch Version:", th.__version__)
