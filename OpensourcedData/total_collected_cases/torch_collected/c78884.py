
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from functools import partial

def train(task, net):
    print(f'task {task} : {list(net.parameters())[0].device}')
    # x = torch.rand(1, 3).cuda(3)
    # y = net(x)
    # print(y.shape)


def main():
    mp.set_start_method('spawn')
    net = nn.Linear(3, 5).cuda(3)
    net = net.share_memory()
    p_train = partial(train, net=net)
    
    with mp.Pool(
        processes=8,
        maxtasksperchild=1,
    ) as pool:
        pool.map(p_train, list(range(16)), chunksize=1)
    

if __name__ == '__main__':
    main()


I tried to make the network global and use `fork` starting method, then the GPU memory usage is normal (takes about ~1GB, which is equal to the program with the same network but without any multiprocessing). However, I couldn't create CUDA tensors inside sub-processes with `fork`, which makes this method usefulness for most of the cases (after all we need to compute something rather than just print it out).


import torch
import torch.nn as nn
import torch.multiprocessing as mp
from functools import partial

net: nn.Module

def train(task):
    global net
    print(f'task {task} : {net} {list(net.parameters())[0].device}')
    x = torch.rand(1, 3).cuda(3) # not allowed
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    y = net(x)
    print(y.shape)


def main():
    mp.set_start_method('fork')
    global net
    net = nn.Linear(3, 5).cuda(3)
    # net = net.share_memory()
    p_train = partial(train)#, net=net)
    
    with mp.Pool(
        processes=8,
        maxtasksperchild=1,
    ) as pool:
        pool.map(p_train, list(range(16)), chunksize=1)
    

if __name__ == '__main__':
    main()
