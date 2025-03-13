
 
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import time
input = torch.rand([64,3,224,224])
conv2d = torch.nn.Conv2d(3,32,(3,3),stride= (2,2), padding = (1,1),bias= False)
time_sort_avg = 0.0
time_conv_avg = 0.0
for i in range(2):
    print(i)
    start =time.time()
    output = conv2d(input)
    time_conv = time.time() -start
    output = output.view(-1)
    start =time.time()
    sorted,index = func_cls(output)
    time_sort = time.time() -start
    time_sort_avg += time_sort
    time_conv_avg += time_conv
r_e_s=time_sort_avg/2

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)