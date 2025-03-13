
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = func_cls(1, 1, (1,), stride=511, padding=255, dilation=(1,))
        self.conv2 = func_cls(1, 1, (1,), stride=2, dilation=(1014,), padding=0)
        self.pool = nn.AvgPool2d(kernel_size=(511,511), stride=3, padding=255)

    def forward(self, i0):
        x = torch.max(i0, dim=3)[0]
        x = self.conv1(x)
        o0 = self.conv2(x)
        o1 = self.pool(x)
        return o0, o1

inp = {
'i0':torch.zeros((1,1,1,1),dtype=torch.float32),
}

mod = MyModule()

out = mod(**inp)
print(f'{out}')

exported = torch.jit.trace(mod, list(inp.values()))
exported = torch.jit.optimize_for_inference(exported)

eout = exported(**inp)
print(f'{eout}')
