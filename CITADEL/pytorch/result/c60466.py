
from contextlib import contextmanager
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import gc
import time
import torch
from torch.nn import functional as F
torch.manual_seed(0)

class Result:
    pass

def unfold1d(input, kernel_size: int, stride: int):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = input[..., :tgt_length].contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    out = input.as_strided(shape + [n_frames, kernel_size], strides)
    return out.transpose(-1, -2)


def torch_unfold(x, kernel, stride):
    B, C, T = x.shape
    frames = func_cls(x[:, :, None], kernel_size=[1, kernel], stride=[1, stride])
    frames = frames.reshape(B, C, kernel, -1)
    return frames


@contextmanager
def measure():
    gc.collect()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    result = Result()
    begin = time.time()
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result.duration = time.time() - begin
        result.mem = torch.cuda.max_memory_allocated() / 2**20


def compare(kernel, stride):
    print("For", kernel, stride)
    x = torch.randn(1, 1, 160000, device='cuda')
    with measure() as r1:
        frames = unfold1d(x, kernel, stride)

    with measure() as r2:
        frames2 = torch_unfold(x, kernel, stride)
    print(f'time unfold1d / time torch unfold {r1.duration / r2.duration:.4f}')
    return r1.duration / r2.duration

# compare(64, 8)
# compare(1024, 256)
# compare(1024, 167)
r_e_s=compare(2048, 190)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)