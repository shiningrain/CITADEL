
from torch import rand
from guppy import hpy

# Get the heap before Tensor creation
hp = hpy()
before = hp.heap()

# Create a dummy Tensor
rand = rand((20, 10))
del rand

# Get the after heap and diff
after = hp.heap()
leftover = after - before
print(leftover)


Output:

Partition of a set of 1 object. Total size = 408 bytes.
 Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
     0      1 100      408 100       408 100 types.FrameType


2. Compare to the following output which creates a tensor, saves it to disk, then reads it with `torch.load()` and deletes it:


from torch import rand, save, load
from guppy import hpy

# Create a dummy Tensor and save to disk
rand = rand((20, 10))
save(rand, 'test.pth')
del rand

# Get the heap before torch.load()
hp = hpy()
before = hp.heap()

# Read and del
test_tensor = load('test.pth')
del test_tensor

# Get the after heap and diff
after = hp.heap()
leftover = after - before
print(leftover)
print('\nGet referrer to leftover dict:')
print(leftover[1].byid.referrers.byvia)


Output:

Partition of a set of 2 objects. Total size = 640 bytes.
 Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
     0      1  50      408  64       408  64 types.FrameType
     1      1  50      232  36       640 100 dict (no owner)

Get referrer to leftover dict:
Partition of a set of 1 object. Total size = 416 bytes.
 Index  Count   %     Size   % Cumulative  % Referred Via:
     0      1 100      416 100       416 100 "['Unpickler']", "['_unpickler']", '.__objclass__',
                                             '.__self__', '[0]'


Given that the uncleared `dict` appears to be referred to by the `Unpickler` object, my thought is that this may be related to the wrapper defined [here](https://github.com/pytorch/pytorch/blob/master/torch/serialization.py#L869), but I may be completely wrong.

3. Finally, run this script to show that the memory usage allocated to the `dict` grows with subsequent `torch.load()` operations:


from torch import rand, save, load
from guppy import hpy

# Create a dummy Tensor and save to disk
rand = rand((20, 10))
save(rand, 'test.pth')
del rand

# Get the heap before torch.load()
hp = hpy()
before = hp.heap()

# Read and del
for i in range(1000):
    test_tensor = load('test.pth')
    del test_tensor
del i

# Get the after heap and diff
after = hp.heap()
leftover = after - before
print(leftover)
print('\nGet referrer to leftover dict:')
print(leftover[0].byid.referrers.byvia)


Output:

Partition of a set of 2 objects. Total size = 9720 bytes.
 Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
     0      1  50     9312  96      9312  96 dict (no owner)
     1      1  50      408   4      9720 100 types.FrameType


Note the size increase of the `dict` from 408 -> 9312 between the examples in step 2 and step 3.

## Expected behavior

All memory occupied during/after the `torch.load()` operation should be cleared after the tensor is deleted/goes out of scope.

## Environment


PyTorch version: 1.9.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: macOS 10.15.7 (x86_64)
GCC version: Could not collect
Clang version: 12.0.0 (clang-1200.0.32.27)
CMake version: Could not collect
Libc version: N/A

Python version: 3.8.5 (default, Sep  4 2020, 02:22:02)  [Clang 10.0.0 ] (64-bit runtime)
Python platform: macOS-10.15.7-x86_64-i386-64bit
Is CUDA available: False
CUDA runtime version: No CUDA
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A

Versions of relevant libraries:
[pip3] numpy==1.19.4
[pip3] pytorch-lightning==1.2.10
[pip3] torch==1.9.0
[pip3] torch-cluster==1.5.9
[pip3] torch-geometric==1.7.1
[pip3] torch-scatter==2.0.7
[pip3] torch-sparse==0.6.10
[pip3] torch-spline-conv==1.2.1
[pip3] torchmetrics==0.2.0
[conda] blas                      1.0                         mkl
[conda] mkl                       2019.5                intel_281    intel
[conda] mkl-service               2.3.0            py38h9ed2024_0
[conda] mkl_fft                   1.2.0            py38hc64f4ea_0
[conda] mkl_random                1.1.1            py38h959d312_0
[conda] numpy                     1.19.4                   pypi_0    pypi
[conda] numpy-base                1.19.2           py38hcfb5961_0
[conda] pytorch-lightning         1.2.10                   pypi_0    pypi
[conda] torch                     1.9.0                    pypi_0    pypi
[conda] torch-cluster             1.5.9                    pypi_0    pypi
[conda] torch-geometric           1.7.1                    pypi_0    pypi
[conda] torch-scatter             2.0.7                    pypi_0    pypi
[conda] torch-sparse              0.6.10                   pypi_0    pypi
[conda] torch-spline-conv         1.2.1                    pypi_0    pypi
[conda] torchmetrics              0.2.0                    pypi_0    pypi


## Additional context

I previously brought this issue up on the [Pytorch forum](https://discuss.pytorch.org/t/uncollected-object-references-from-torch-load-cause-memory-growth/130188) and will include the relevant text below:

I’m currently working on training a model on a rather large dataset ( ~1 billion examples in total, with multiple individual tensors that need to be loaded per each example). To store and load this dataset, I’m employing the [Webdataset library](https://github.com/webdataset/webdataset). I am storing preprocessed examples as Pytorch tensors in a tar file, per the Webdataset spec. I am running into a memory issue after a long period of training where the machine performing the reading/loading operation runs out of memory. After a bit of debugging, it seems to be caused by some unexpected (at least to me) behavior with the Pytorch serialization code.

Since each tensor is stored as a separate file, then tarred together, each must be loaded individually, meaning that ~1 billion torch.load() operations take place per epoch. I traced memory usage using the tracemalloc utility and discovered that some part of this serialization process is creating objects that are not garbage collected even after the tensor read from disk goes out of scope (or in the example below, is manually deleted).

I have included a minimal example below to show these objects. The example creates a dummy tensor, then reads it into memory repeatedly. The memory usage is queried in 10,000 step intervals. Given that that the tensor goes out of scope immediately after it is read with torch.load(), I would not expect to see any objects from torch/serialization.py to still be in scope. However, I instead see ~10,000 new objects created each time the memory is queried (every 10,000 steps, see example output below). These objects also seem to never be released and cause constant linear memory growth. Even though they are only 28B each, on average, they still seem to be causing memory use to grow linearly until the machine is out of memory (usually after many millions of examples are read from disk on a high-mem training machine).

Thank you in advance for any and all help you can offer! I sincerely apologize if this has been addressed elsewhere and I missed it while searching. Right now this is blocking bug for one of my ongoing projects.


import torch
import tracemalloc

if __name__ == '__main__':
    # Create a dummy Tensor serialized to disk
    rand = torch.rand((20, 10))
    torch.save(rand, 'test.pth')

    # Start tracemalloc
    tracemalloc.start(30)
    old_snapshot = tracemalloc.take_snapshot()

    for i in range(30001):
        # Load the Tensor
        test_tensor = torch.load('test.pth')

        # Immediately delete reference to test_tensor
        del test_tensor

        if i % 10000 == 0 and i != 0:
            # Take snapshot
            snapshot = tracemalloc.take_snapshot()

            # Print changes in memory consumption
            print(f'################# STEP {i} #################')
            for stat in snapshot.compare_to(old_snapshot, 'lineno')[:2]:
                print(str(stat))
            print('############################################')

            # Save snapshot
            old_snapshot = snapshot

