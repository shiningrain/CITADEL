
#!/usr/bin/env python3

import torch
from torch import nn
from torchaudio.models import Conformer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_dim = 272
        num_heads = 8
        num_layers = 18
        conv_kernel_size = 31
        num_labels = 500

        conv_layers = [
            nn.Conv1d(
                in_channels=80, out_channels=encoder_dim * 2,
                kernel_size=3, stride=1, padding=0, dilation=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=encoder_dim, out_channels=encoder_dim * 2,
                kernel_size=3, stride=2, padding=0, dilation=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=encoder_dim, out_channels=encoder_dim * 2,
                kernel_size=3, stride=1, padding=0, dilation=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=encoder_dim, out_channels=encoder_dim * 2,
                kernel_size=3, stride=2, padding=0, dilation=1),
            nn.GLU(dim=1),
        ]

        self.conv_subsampler = nn.Sequential(*conv_layers)

        self.conformer = Conformer(
            input_dim=encoder_dim, num_heads=num_heads, ffn_dim=4 * encoder_dim,
            num_layers=num_layers, depthwise_conv_kernel_size=conv_kernel_size,
        )

        self.output_layer = nn.Linear(encoder_dim, num_labels + 1)

    def forward(self, input_t: torch.Tensor):
        output = self.conv_subsampler(input_t.unsqueeze(1).transpose(2, 3).squeeze(1)).transpose(1, 2)
        output, output_lengths = self.conformer(output, torch.tensor([output.shape[1]]).cuda())
        output = self.output_layer(output)
        return output, output_lengths


# 249 or 250 here causes "RuntimeError: CUDA error: an illegal memory access was encountered"
input_length = 250

input_tensor = torch.rand((1, input_length, 80), dtype=torch.float32).cuda()
model = Model().cuda()
output, output_len = model(input_tensor)
output = nn.functional.log_softmax(output, dim=2).transpose(0, 1)
lossfun = nn.CTCLoss(blank=0, reduction='mean')
print(output.shape)
loss = lossfun(output, torch.cuda.IntTensor([[1, 2]]), output_len, torch.cuda.IntTensor([2]))
loss.backward()
