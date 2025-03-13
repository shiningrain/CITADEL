
import time
import torch as th

N = 100
X = th.randn(153531, 4, 4, device="cuda")

th.inverse(X)

th.cuda.synchronize()
start = time.time()

for _ in range(N):
    th.inverse(X)

th.cuda.synchronize()
end = time.time()

print("Time per inverse (ms):", 1000 * (end - start) / N)
print("PyTorch Version:", th.__version__)
