
import torch
import torch.nn.functional as F

F.avg_pool2d(torch.view_as_complex_copy(torch.randn([1,1,32,32,2])),2)
