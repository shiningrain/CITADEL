
import torch
results = dict()
input = torch.rand([2, 1])
torch.fft.irfftn(input)
# RuntimeError: out_size == signal_size[i + 1] || out_size == (signal_size[i + 1] / 2) + 1INTERNAL ASSERT FAILED at "../aten/src/ATen/native/mkl/SpectralOps.cpp":458, please report a bug to PyTorch
# torch.fft.irfft2(input)
# # RuntimeError: out_size == signal_size[i + 1] || out_size == (signal_size[i + 1] / 2) + 1INTERNAL ASSERT FAILED at "../aten/src/ATen/native/mkl/SpectralOps.cpp":458, please report a bug to PyTorch.
