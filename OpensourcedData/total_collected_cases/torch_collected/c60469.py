
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import mkl
import omp_thread_count

def main():
    print(mkl.get_max_threads(), torch.get_num_threads(), omp_thread_count.get_thread_count())
    data = FakeData(transform=ToTensor())
    dataloader = DataLoader(data, num_workers=4, pin_memory=True)
    print(mkl.get_max_threads(), torch.get_num_threads(), omp_thread_count.get_thread_count())
    for e in range(3):
        print(f'epoch {e}:')
        for _ in dataloader:
            pass
    print(mkl.get_max_threads(), torch.get_num_threads(), omp_thread_count.get_thread_count())

if __name__ == '__main__':
    main()
