import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
pst=pdb.set_trace
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
device = 'cuda:0'
convx = nn.Conv2d(1, 100, (5,128)).to(device)
torch.cuda.empty_cache()
y = convx(torch.randn(64, 1, 128, 128).to(device))
