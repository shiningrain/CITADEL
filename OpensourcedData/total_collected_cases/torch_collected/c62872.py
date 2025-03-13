
import torch
import torch.nn as nn
import torch.utils.mobile_optimizer as mobile_optimizer
import torch.nn.functional as F

class Demo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        return x

model = Demo()
model = torch.quantization.convert(model)
model = torch.jit.script(model)
model = mobile_optimizer.optimize_for_mobile(model, backend='Metal')
model._save_for_lite_interpreter('model.ptl')

x = torch.rand((1, 3, 256, 256))
out = model(x)
print(out.shape)

