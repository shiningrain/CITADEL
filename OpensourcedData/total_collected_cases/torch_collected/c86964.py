import torch
from torch import Tensor
from torch.utils.benchmark import Timer, Compare
from itertools import product


def custom_lerp(x: Tensor, y: Tensor, ratio: float) -> Tensor:
    return y.mul_(1.0 - ratio).add_(x, alpha=ratio)


def lerp(x: Tensor, y: Tensor, ratio: float) -> Tensor:
    return y.lerp_(x, ratio)


def gen_inputs():
    shapes = ((3, 400, 400),)
    devices = ("cpu", "cuda")
    fns = (lerp, custom_lerp)
    for shape, device, fn in product(shapes, devices, fns):
        t1 = torch.randn(shape, device=device)
        t2 = torch.randn(shape, device=device)
        assert torch.allclose(fns[0](t1, t2, 0.3), fns[1](t1, t2, 0.3))
        yield f"lerp {device} {t1.dtype}", str(tuple(shape)), 1, fn, t1, t2, 0.3


def benchmark(label, sub_label, threads, f, *args, **kwargs):
    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label=label,
                 description=f.__name__,
                 sub_label=sub_label,
                 num_threads=threads).blocked_autorange()


results = []
for args in gen_inputs():
    results.append(benchmark(*args))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()
