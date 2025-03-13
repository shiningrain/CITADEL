
import torch
import torch.nn as nn

err_num_a = 0
err_num_b = 0
correct_num = 0
for _ in range(100000):
    x = torch.rand((1, 1, 28, 28)).clip(0, 1)
    conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1))

    a = torch.mean(conv_1(x))

    with torch.no_grad():
        # torch.set_grad_enabled(False)
        b = torch.mean(conv_1(x))

    if torch.isinf(a) or torch.isnan(a):
        err_num_a += 1
        print(a, b)
    if torch.isinf(b) or torch.isnan(b):
        err_num_b += 1
        print(a, b)
    else:
        correct_num += 1

print(f"correct num = {correct_num} err_num_a = {err_num_a} err_num_b = {err_num_b}")

