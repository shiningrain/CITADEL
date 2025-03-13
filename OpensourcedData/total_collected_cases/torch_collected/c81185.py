
# bug_demo.py

import torch

n_trials = 100
for ii in range(n_trials):
    a = torch.randn(1024, device='mps')
    b = torch.randn(499, device='mps')
    c = torch.nn.functional.conv1d(a.view(1, 1, -1), b.view(1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'mps: trial {ii}, nan elements {torch.isnan(c.squeeze()).nonzero().view(-1).cpu().numpy()}')
        
for ii in range(n_trials):
    a = torch.randn(1024, device='cpu')
    b = torch.randn(499, device='cpu')
    c = torch.nn.functional.conv1d(a.view(1, 1, -1), b.view(1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'cpu: trial {ii}, elements {torch.isnan(c.squeeze()).nonzero().view(-1).numpy()}')
