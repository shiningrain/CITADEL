
import torch

# Loads one set of parameters into memory.
m = MyModule()

# Loads a second set of parameters into memory.
state_dict = torch.load('foo.pt')

m.load_state_dict(state_dict)


There should be an officially-supported way to load a module with only one set of parameters in memory at once.

## Pitch

One way to avoid the extra copy of the parameters is to instantiate the module with `device='meta'` so as not to allocate any storage.

import torch

m = MyModule(device='meta')
state_dict = torch.load('foo.pt')
m.load_state_dict(state_dict)

# Problem: module parameters are still meta tensors; loading didn't work.


However, this currently doesn't work because `load_state_dict()` copies data into  the module parameters when loading; the module's parameters will still be on the meta device after the call to `load_state_dict()`.

One way to address this is to implement a flag that "reuses" the state_dict tensors passed in to `load_state_dict()`. With this set, the underlying data for module parameters will be swapped out to reference the same memory as the state_dict tensors. **Note that this would also require `set_data()` to be implemented for meta tensors, allowing them to be converted to non-meta tensors.**

Thus, only one copy of the module parameters needs to exist in memory for the module to be loaded. Also, since the parameters remain the same objects, optimizers that reference them will still work.

import torch

m = MyModule(device='meta')
optimizer = torch.optim.SGD(m.parameters(), lr=0.01)

state_dict = torch.load('foo.pt')
m.load_state_dict(state_dict, reuse_tensors=True)

# Module parameters now reference the same data as state_dict tensors.
# The optimizer defined above will still correctly operate on the
# module's parameters.
