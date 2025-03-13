
import numpy as np
import torch 
import torch.nn as nn

tmp_log = np.load("./tmp_log.npy")
labels = np.load("./proj_labels.npy")

log = torch.from_numpy(tmp_log).cuda()
labels = torch.from_numpy(labels).long().cuda()
loss = nn.NLLLoss().cuda()

for i in range(50):
    output = loss(log, labels)
    print(i, output.item())

