
import torch as th
import numpy as np

total_trials = 10000
failures_torch_cpu = 0
failures_torch_cuda = 0
failures_numpy = 0
failure_threshold = 1e-1

for i in range(total_trials):
    M = th.cat(
    [th.cat([th.diag(th.rand(3)*4), ((th.rand(3) - 0.5)*10000)[:, None]], dim=1), 
       th.tensor([[0, 0, 0, 1]], dtype=th.float)], dim=0)
    if (th.pinverse(M) @ M - th.eye(4).to(M)).abs().max() > failure_threshold:
        failures_torch_cpu += 1    
    M = M.cuda()
    if (th.pinverse(M) @ M - th.eye(4).to(M)).abs().max() > failure_threshold:
        failures_torch_cuda += 1
    if np.abs(np.linalg.pinv(M.cpu().numpy()) @ M.cpu().numpy() - np.eye(4)).max() > failure_threshold:
        failures_numpy += 1
        
print(f"""
Failures torch_cpu: {failures_torch_cpu / total_trials}.
Failures torch_cuda: {failures_torch_cuda / total_trials}.
Failures numpy: {failures_numpy / total_trials}.""")
