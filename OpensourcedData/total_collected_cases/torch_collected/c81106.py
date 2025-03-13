
import gc

import torch
import torchvision


def bench(model, device, memory_format):

    model = model.to(device=device, dtype=torch.float, memory_format=memory_format, non_blocking=False)
    model.eval()

    batch = torch.rand(8,3,256,256, device=device).to(device=device, dtype=torch.float, memory_format=memory_format, non_blocking=False)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.inference_mode():
        with torch.autocast(device.type, enabled=True):
            model(batch)

    del batch



class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, bias=False)


    def forward(self, x):
        print('before', x.device, x.dtype, x.layout, x.is_contiguous())
        x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
        print('after', x.device, x.dtype, x.layout, x.is_contiguous())

        x = self.conv1(x)

        return x


def main_worker():

    device = torch.device("cuda")


    bench_model = MyModel()
    bench_model.eval()

    bench(bench_model, device, torch.contiguous_format)
    bench(bench_model, device, torch.torch.channels_last)


if __name__ == '__main__':

    main_worker()


Output:

before cuda:0 torch.float32 torch.strided True
after cuda:0 torch.float32 torch.strided True
before cuda:0 torch.float32 torch.strided False
after cuda:0 torch.float32 torch.strided True
Traceback (most recent call last):
  File "1.py", line 57, in <module>
    main_worker()
  File "1.py", line 52, in main_worker
    bench(bench_model, device, torch.torch.channels_last)
  File "1.py", line 20, in bench
    model(batch)
  File "/home/xxxxx/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "1.py", line 38, in forward
    x = self.conv1(x)
  File "/home/xxxxx/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/xxxxx/.local/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 457, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home/xxxxx/.local/lib/python3.8/site-packages/torch/nn/modules/conv.py", line 453, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
You can try to repro this exception using the following code snippet. If that doesn't trigger the error, please include your original repro script when reporting this issue.


import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([8, 1, 256, 256], dtype=torch.half, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(1, 8, kernel_size=[5, 5], padding=[0, 0], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().half()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

ConvolutionParams
    data_type = CUDNN_DATA_HALF
    padding = [0, 0, 0]
    stride = [1, 1, 0]
    dilation = [1, 1, 0]
    groups = 1
    deterministic = false
    allow_tf32 = true
input: TensorDescriptor 0x4f4374d0
    type = CUDNN_DATA_HALF
    nbDims = 4
    dimA = 8, 1, 256, 256,
    strideA = 65536, 1, 256, 1,
output: TensorDescriptor 0x4f448120
    type = CUDNN_DATA_HALF
    nbDims = 4
    dimA = 8, 8, 252, 252,
    strideA = 508032, 1, 2016, 8,
weight: FilterDescriptor 0x4f1b6720
    type = CUDNN_DATA_HALF
    tensor_format = CUDNN_TENSOR_NHWC
    nbDims = 4
    dimA = 8, 1, 5, 5,
Pointer addresses:
    input: 0x7f9abe800800
    output: 0x7f9abf800000
    weight: 0x7f9abe800000
Forward algorithm: 1
