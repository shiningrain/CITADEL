import torch

# Fails despite being able to work correctly
conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), \
    stride=(2, 2), dilation=(1, 1), groups=1, bias=True, padding="same")

