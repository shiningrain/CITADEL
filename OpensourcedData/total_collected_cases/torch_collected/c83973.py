py
import torch 

def foo(idx):
    print(idx)
    torch.cuda.set_device(idx)

torch.cuda.device_count()  # this call causes an error: "Cannot re-initialize CUDA in forked subprocess"
assert not torch.cuda.is_initialized()
torch.multiprocessing.start_processes(foo, nprocs=2, start_method="fork")
