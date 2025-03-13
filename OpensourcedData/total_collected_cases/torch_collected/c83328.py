
results = dict()
import torch
arg_class = torch.nn.Conv2d(512,2048,1)
arg_1 = torch.rand([128, 512, 16, 16], dtype=torch.float16)
results["time_low"] = arg_class(arg_1)
