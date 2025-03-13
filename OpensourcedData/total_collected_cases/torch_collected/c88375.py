import torch
import functorch
from torch._dynamo.optimizations.training import aot_nvprims_nvfuser, aot_nvprims_aten
from torch._prims.context import TorchRefsNvfuserCapabilityMode
optimize = torch._dynamo.optimize(aot_nvprims_aten)

torch.cuda.cudart().cudaProfilerStart()

class Fusion(torch.nn.Module):
  def __init__(self) :
    super(Fusion, self).__init__()
    self.conv = torch.nn.Conv2d(32, 32, (1, 1), bias=False)
    self.norm = torch.nn.InstanceNorm2d(32, track_running_stats=True)

  def forward(self, inp) :
    out = self.conv(inp)
    out = out.relu()
    out = self.norm(out)
    return out

model = Fusion().cuda()

input1 = torch.randn(2, 32, 8, 8, device="cuda")

optimized_model = optimize(model)
with torch.cuda.amp.autocast(False):
  for _ in range(5):
    out = optimized_model(input1)
    out.sum().backward()

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
