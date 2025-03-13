

# functorch.__version__
# '1.13.0+cu117'

import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import functorch

jacobian = functorch.jacrev

x = torch.tensor([0.0, 0.0], dtype=torch.float64)
y = torch.tensor([0.0, 0.0], dtype=torch.float64)

def foo(x, y):
    x1, x2 = x
    y1, y2 = y
    return torch.stack([x1*x2*y1*y2])

def bar1(x, y):
    power = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    state = torch.cat([x, y])
    state = func_cls(state, power)
    return torch.stack([state.prod(-1)])

def bar2(x, y):
    power = [1, 1, 1, 1]
    # power = torch.tensor([1, 1, 1, 1], dtype=torch.int64) # -- will also break
    state = torch.cat([x, y])
    state = torch.stack([s**p for s, p in zip(state, power)])
    return torch.stack([state.prod(-1)])

print(foo(x, y).numpy())
print(bar1(x, y).numpy())
print(bar2(x, y).numpy())

# d^2(d^2(f)/d(y^2))/d(x^2)

print(jacobian(jacobian(lambda x: jacobian(jacobian(lambda y: foo(x, y)))(y)))(x).flatten().numpy())
r_e_s=jacobian(jacobian(lambda x: jacobian(jacobian(lambda y: bar1(x, y)))(y)))(x).flatten().numpy()

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)

