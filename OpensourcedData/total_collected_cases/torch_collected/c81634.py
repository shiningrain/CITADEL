
import time

import torch

lstm = torch.nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

inp = torch.rand(8, 2000, 256)
h0 = torch.rand(2, 8, 256)
c0 = torch.rand(2, 8, 256)

tick = time.time()
_ = lstm(inp, (h0, c0))
tock = time.time()
print(tock - tick)
