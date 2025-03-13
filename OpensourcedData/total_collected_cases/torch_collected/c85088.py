
import torch
from torch.utils.checkpoint import checkpoint

from collections import namedtuple

Tup = namedtuple("Tup", "a b c")

tup = Tup(torch.ones(10, requires_grad=True), torch.ones(10, requires_grad=True), torch.ones(10, requires_grad=True))

def foo(tup):
	return Tup(tup.a + tup.b, tup.b, tup.a + tup.b + tup.c)

import pdb ; pdb.set_trace()
out = checkpoint(foo, tup, use_reentrant=False)
