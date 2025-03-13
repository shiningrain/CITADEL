bash
> nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3090 
GPU 1: NVIDIA GeForce RTX 3090 
GPU 2: NVIDIA TITAN RTX
GPU 3: Quadro GV100
GPU 4: NVIDIA RTX A6000
GPU 5: NVIDIA TITAN RTX
GPU 6: NVIDIA RTX A6000


Until version 1.11, I used the following code to switch GPUs.

python
import os
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

device = torch.device("cuda:0")


Output: 

bash
Using GPU is CUDA:1
CUDA:0 NVIDIA GeForce RTX 3090, 24268.3125MB


Since version 1.12, the output has changed to the following.

bash
Using GPU is CUDA:1
CUDA:0 NVIDIA RTX A6000, 48685.3125MB
CUDA:1 NVIDIA RTX A6000, 48685.3125MB
CUDA:2 NVIDIA GeForce RTX 3090, 24268.3125MB
CUDA:3 NVIDIA GeForce RTX 3090, 24268.3125MB
CUDA:4 Quadro GV100, 32508.375MB
CUDA:5 NVIDIA TITAN RTX, 24220.4375MB
CUDA:6 NVIDIA TITAN RTX, 24220.4375MB


This may be due to a change in the timing of reading environment variables.

GPU allocation to processes by the environment variable CUDA_VISIBELE_DEVICES does not seem to be working.

I tried the following code for loading the torch module, but did not get the expected behavior as in version 1.11.

python
import os
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

# print using GPU Info
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")


and `del torch`

python
import os
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

del torch
import torch

# print using GPU Info
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")


and `importlib.reload()`

python
import os
import torch
import importlib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

importlib.reload(torch)

# print using GPU Info
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")

Even with version 1.12, I can switch GPUs by using the following code.

python
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

# print using GPU Info
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")
