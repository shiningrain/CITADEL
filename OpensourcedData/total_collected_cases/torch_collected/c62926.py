## To Reproduce


import torch
device = torch.device('cuda')
dtype = torch.float
inp = torch.rand(1, 1, 8, 8, dtype=dtype, device=device)
w = torch.randn(1, 1, 1, 1, dtype=dtype, device=device)
conv2d_out = torch.conv2d(inp, w, None, (1, 1), (0, 0), (1, 1), 1)
print(conv2d_out.size())
print(inp.size())
cudnn_out = torch.cudnn_convolution_add_relu(inp, w, inp, 1.0, None, (1, 1), (0, 0), (1, 1), 1)
torch.cuda.synchronize()
