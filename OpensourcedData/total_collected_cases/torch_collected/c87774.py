
# # reproduce case 1
# import torch

# # the range argument matters here. (-4, 5) and (-7, 8) raises exception but (-3, 4) runs fine.
# coords = torch.arange(-7, 8, dtype=torch.float32)
# grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
# abs_output = torch.abs(grid) + 1.0
# print(torch.log2(abs_output)) # FPE here



# reproduce case 2
import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, 2, batch_first=True)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 5, 20)
c0 = torch.randn(2, 5, 20)
output, (hn, cn) = rnn(input, (h0, c0)) # FPE here
