
import torch
import time
from torch import einsum

# MPS backend has no synchronization API comparable to CUDA's:
# https://pytorch.org/docs/master/notes/cuda.html#asynchronous-execution
# so we have to get creative.
# to force Python to wait for computation to complete: introduce a data-dependency on every element in the tensor.
baseline_start = time.perf_counter()
torch.rand(16, 4096, 4096, dtype=torch.float, device="mps").max().item()
print('for reference: our sync operation -- on a tensor of our target size -- is responsible for only %.4f seconds of overhead' % (time.perf_counter()-baseline_start))

repeats = 10

ein0_batch_duration = 0
for ix in range(repeats):
  q = torch.rand(16, 4096, 40, dtype=torch.float, device="mps")
  k = torch.rand(16, 4096, 40, dtype=torch.float, device="mps")
  ein0_start = time.perf_counter()
  einsum('b i d, b j d -> b i j', q, k).max().item()
  ein0_duration = time.perf_counter()-ein0_start
  print('einsum 0 iteration %d took %.4f seconds' % (ix, ein0_duration))
  ein0_batch_duration += ein0_duration
print('%d iterations of einsum 0 took %.4f seconds; avg %.4f secs' % (repeats, ein0_batch_duration, ein0_batch_duration/repeats))

ein1_batch_duration = 0
for ix in range(repeats):
  attn = torch.rand(16, 4096, 4096, dtype=torch.float, device="mps")
  v = torch.rand(16, 4096, 40, dtype=torch.float, device="mps")
  ein1_start = time.perf_counter()
  einsum('b i j, b j d -> b i d', attn, v).max().item()
  ein1_duration = time.perf_counter()-ein1_start
  print('einsum 1 iteration %d took %.4f seconds' % (ix, ein1_duration))
  ein1_batch_duration += ein1_duration
print('%d iterations of einsum 1 took %.4f seconds; avg %.4f secs' % (repeats, ein1_batch_duration, ein1_batch_duration/repeats))
