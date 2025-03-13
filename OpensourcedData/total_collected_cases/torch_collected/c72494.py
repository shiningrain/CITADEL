
import torch
import torch.nn as nn
import onnx


class BatchNorm1dNoAutoCast(nn.BatchNorm1d):
    def init(self, num_features, **kwargs):
        nn.BatchNorm1d.init(self, num_features, **kwargs)

    def forward(self, x):
        if False:  # torch.onnx.is_in_onnx_export():
            ret = nn.BatchNorm1d.forward(self, x.to(torch.float)).to(x.dtype)
        else:
            ret = nn.BatchNorm1d.forward(self, x)
        return ret


mod = BatchNorm1dNoAutoCast(5).to(device="cuda").eval()

data = torch.rand(300, 5, device="cuda").half()
with torch.inference_mode(), torch.cuda.amp.autocast():
    torch.onnx.export(mod,
                      (data),
                      "a.onnx",
                      opset_version=14,
                      verbose=True
                      )

onnx_model = onnx.load("a.onnx")
onnx.checker.check_model(onnx_model, full_check=True)
