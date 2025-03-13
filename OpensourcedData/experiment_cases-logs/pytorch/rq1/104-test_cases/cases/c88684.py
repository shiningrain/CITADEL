
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import tempfile

model = torch.nn.Linear(3, 1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
lr = func_cls(optimizer, 0.1, 1.0)

tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    torch.save(lr.state_dict(), f)
