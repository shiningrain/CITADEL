
import torch
xx = torch.zeros(3, 4)
xx.foo = 'bar'
torch.save(xx, './_xx.pt')
torch.load('./_xx.pt').foo
