
from torch import nn

a = [nn.Module()]
b = [nn.Module()]
c = a + b
print(c)

a = nn.ModuleList([nn.Module()])
b = nn.ModuleList([nn.Module()])
c = a + b # TypeError: unsupported operand type(s) for +: 'ModuleList' and 'ModuleList'
print(c)
