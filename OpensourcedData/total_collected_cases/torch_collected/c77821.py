
import torch.nn as nn
import torch

cpu_device = torch.device("cpu")


class Hybrid_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1, bias=False)
    self.layer2 = nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1, bias=False)

  def forward(self, inputs):
    x = self.layer1(inputs)
    x = torch.reshape(x, (x.size(0), x.size(1), x.size(2), 16, 2))
    x = self.layer2(x)
    return x


if __name__ == "__main__":
  test_model = Hybrid_model().to(memory_format=torch.channels_last)
  x = torch.randn([3, 2, 32, 32], dtype=torch.float, requires_grad=True)
  y = test_model(x)
