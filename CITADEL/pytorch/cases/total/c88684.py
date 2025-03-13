
import torch
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
lr = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1.0)

tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    torch.save(lr.state_dict(), f)
