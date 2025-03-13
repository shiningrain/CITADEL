# A generated case that leads to the failure in matching Conv2d and LazyConvTranspose2d (neigher status equal nor value equal)
import torch
results = dict()
in_channels = 2048
out_channels = 192
kernel_size = 1
bias = False
input_signature_0_tensor = torch.rand([80, 2048, 6, 6], dtype=torch.float32)
input_signature_0 = input_signature_0_tensor.clone()
input_signature = [input_signature_0,]
try:
  arg_class = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, )
  results["res_1"] = arg_class(*input_signature)
except Exception as e:
  results["err_1"] = "ERROR:"+str(e)
try:
  results["res_2"] = torch.nn.LazyConvTranspose2d(in_channels,kernel_size,output_padding=out_channels,bias=bias,)(*input_signature)
except Exception as e:
  results["err_2"] = "ERROR:"+str(e)

print(results)
