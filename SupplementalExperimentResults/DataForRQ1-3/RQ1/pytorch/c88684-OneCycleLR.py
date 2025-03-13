
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
lr = func_cls(optimizer, max_lr=1.0, base_momentum=0.8, max_momentum=0.9)

tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    torch.save(lr.state_dict(), f)
