
import math
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import numpy as np
import time
import torch

np.random.seed(42)
torch.manual_seed(0)

torchtime, nptime = 0,0
lintorch = torch.linspace(0, 1, 40000)

for i in range(64):
    start = time.time()
    torchout =  func_cls(lintorch * math.pi / 2 + math.pi / 2)
    torchtime+=(time.time()-start)

    start = time.time()
    npout =  func_cls(lintorch.numpy() * math.pi / 2 + math.pi / 2)
    nptime+=(time.time()-start)

r_e_s=torchtime-nptime

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)