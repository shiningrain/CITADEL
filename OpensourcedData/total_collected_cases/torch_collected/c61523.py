
import torch
import torch.nn.functional as F
from timeit import default_timer as timer

def _compare_cuda(log_probs: torch.Tensor, targets: torch.Tensor):
    # Compute and time manual reduction
    start_manual = torch.cuda.Event(enable_timing=True)
    end_manual = torch.cuda.Event(enable_timing=True)

    start_manual.record()
    loss_manual = F.nll_loss(
        input=log_probs,
        target=targets,
        reduction='none'
    )
    loss_manual = loss_manual.mean()
    end_manual.record()

    # Compute and time auto reduction
    start_auto = torch.cuda.Event(enable_timing=True)
    end_auto = torch.cuda.Event(enable_timing=True)

    start_auto.record()
    loss_auto = F.nll_loss(
        input=log_probs,
        target=targets,
        reduction='mean'
    )
    end_auto.record()

    # Calculate times
    torch.cuda.synchronize()
    time_manual_ms = start_manual.elapsed_time(end_manual)
    time_auto_ms = start_auto.elapsed_time(end_auto)

    return loss_manual, time_manual_ms, loss_auto, time_auto_ms

def _compare_cpu(log_probs: torch.Tensor, targets: torch.Tensor):
    # Compute and time manual reduction
    start_manual = timer()
    loss_manual = F.nll_loss(
        input=log_probs,
        target=targets,
        reduction='none'
    )
    loss_manual = loss_manual.mean()
    time_manual_ms = (timer() - start_manual) * 1000

    # Compute and time auto reduction
    start_auto = timer()
    loss_auto = F.nll_loss(
        input=log_probs,
        target=targets,
        reduction='mean'
    )
    time_auto_ms = (timer() - start_auto) * 1000

    return loss_manual, time_manual_ms, loss_auto, time_auto_ms

def compare_nll_loss(device: str, logits_type: torch.dtype) -> None:
    torch.manual_seed(2021)

    # Reproduced example from DeepLabV3-ResNet50, on PascalVOC 2012
    N, C, H, W = 64, 21, 350, 350
    logits_mean = 0.0066
    logits_std = 0.0487

    # Create targets with shape (N, H, W) and values between [0, C-1] for C classes
    targets = torch.randint(high=C, size=(N, H, W), device=device, dtype=torch.int64)

    # Create logits with shape (N, C, H, W) for C classes
    logits = torch.normal(mean=logits_mean, std=logits_std, size=(N, C, H, W))
    logits = logits.type(logits_type).to(device)
    log_probs = F.log_softmax(logits, dim=1)

    if 'cuda' in device:
        loss_manual, time_manual_ms, loss_auto, time_auto_ms = \
            _compare_cuda(log_probs=log_probs, targets=targets)
    else:
        loss_manual, time_manual_ms, loss_auto, time_auto_ms = \
            _compare_cpu(log_probs=log_probs, targets=targets)

    print('-' * 60)
    print(f'Comparison with device={device}, dtype={logits_type}')
    print('-' * 60)
    print(f'Manual reduction ({time_manual_ms:.3f}ms): {loss_manual}')
    print(f'Auto reduction ({time_auto_ms:.3f}ms):   {loss_auto}')
    print('-' * 60)

if __name__ == '__main__':
    # CPU tests (16-bit not supported)
    compare_nll_loss(device='cpu', logits_type=torch.float64)
    compare_nll_loss(device='cpu', logits_type=torch.float32)

    # GPU tests
    compare_nll_loss(device='cuda:0', logits_type=torch.float64)
    compare_nll_loss(device='cuda:0', logits_type=torch.float32)
    compare_nll_loss(device='cuda:0', logits_type=torch.float16)
