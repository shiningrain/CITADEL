
import torch
from torch import nn

class WeightNormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 16, 8, padding=4, groups=2)
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states

torch.set_default_dtype(torch.float16)

model = WeightNormModule()
input_values = torch.ones(2, 16, 4, dtype=torch.float16)

model = model.to("cuda")
input_values = input_values.to("cuda")

out = model(input_values)
