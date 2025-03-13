
from torch import nn
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

print(torch.__version__)

with torch.no_grad():
    x = torch.zeros(1, 3, 640, 640)
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    script_model = torch.jit.trace(model, x)
    optimized_traced = optimize_for_mobile(script_model, backend='vulkan')
    optimized_traced._save_for_lite_interpreter("sample_data/t.ptl")
