
import torch
import math
import random
import numpy as np
import os

# make ourselves as deterministic as possible.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# (deterministic) messy number generator
v = torch.Tensor([(x*math.pi)%1 for x in range(1000000)]).to(torch.device("cuda"))

# init a and b to the same cumulative sum.
a = b = v.cumsum(0)
counter = 0

# this loop should never terminate.
while ((a==b).all()):
	b = v.cumsum(0)
	counter+=1

print(f"non-determinism after {counter} runs.")
print("errors:",(a-b).abs()[a-b!=0])
print("first error index:",torch.where((a-b)!=0)[0][0])
