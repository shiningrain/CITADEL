
import numpy as np
import torch
from torch.nn import BatchNorm2d, init

np.random.seed(0)
inputs_2d = np.random.normal(size=(16, 3, 8, 8)).astype(np.float32)

bn_momentum = 0.

offline_bn_torch = BatchNorm2d(3, eps=1e-18, momentum=1-bn_momentum)
init.constant_(offline_bn_torch.weight, 1)
init.constant_(offline_bn_torch.bias, 0)
offline_bn_torch.reset_running_stats()
print(list(offline_bn_torch.buffers())[1])
offline_bn_torch.train()
offline_bn_torch(torch.tensor(inputs_2d))
print(list(offline_bn_torch.buffers())[1])
inputs_2d = np.transpose(inputs_2d, (0, 2, 3, 1))
inputs_2d = np.reshape(inputs_2d, (-1, 3))
print(np.std(inputs_2d, axis=0)**2)
print(torch.var(torch.tensor(inputs_2d), 0, unbiased=True, keepdim=False))
print(torch.var(torch.tensor(inputs_2d), 0, unbiased=False, keepdim=False))
