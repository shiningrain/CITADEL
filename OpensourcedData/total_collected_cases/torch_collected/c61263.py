
import torch
import math

# Size in GB
tensor_size = 4
print("Tensor size ", tensor_size, "GB")
# Change this to ~60% of your RAM size
model_size = 16
print("Model size ", model_size, "GB")
# The number of layer will be adjusted to fit in less than model size
layers_num = model_size // tensor_size
kB = 1024
MB = kB * kB
GB = MB * kB
#Number size
precision_size = 4
activation_size = math.floor(math.sqrt(tensor_size * GB / precision_size))
print("Activation size ", math.floor(activation_size / kB * precision_size), "kB")

class Net(torch.nn.Module):
    def __init__(self, empty_init = False):
        super(Net, self).__init__()
        for i in range(layers_num):
            name = "fc_%d" % i
            linear = torch.nn.Linear(activation_size, activation_size)
            setattr(self, name, linear)
    def forward(self, x):
        for i in range(layers_num):
            name = "fc_%d" % i
            linear = getattr(self, name)
            x = linear(x)
        return x

model = Net()


print("Model created")

input = torch.zeros(activation_size, requires_grad=True)
output = model(input)
print("Output ", output.size())


with torch.no_grad():
    torch.onnx.export(model, (input, ), './model_large.onnx', do_constant_folding=False, opset_version=13,  use_external_data_format=True, enable_onnx_checker=False)

print("ONNX saved")

