
import torch.nn as nn 
import torch

device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')

bs = 32
T = 12
d = 8
e = 10
x = torch.rand(bs, T, d).to(device)
init_hidden = torch.rand(1, bs, e).to(device)

model = nn.GRU(input_size=d, hidden_size=e, num_layers=1, batch_first=True)
model = model.to(device)
output, _ = model(x, init_hidden)
