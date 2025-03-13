
import torch

import pickle

with open('failing_input.pkl', 'rb') as f:
    data = pickle.load(f)

args = data['batched_arg']
kwargs = data['kwarg_values']  # groups = 4

i = args[0]  # torch.Size([2, 4, 6, 6])
w = args[1]  # torch.Size([4, 1, 3, 3])
b = args[2][:, 0]  # torch.Size([4, 2]) (bias is batched)

o = torch.nn.functional.conv2d(i, w, b, **kwargs)

print(o[0])
