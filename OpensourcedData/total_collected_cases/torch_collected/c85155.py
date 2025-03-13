
import torch
layer = torch.nn.PixelShuffle(1)

model_input = torch.ones((1,1,1,1,0))
pred = layer(model_input)
