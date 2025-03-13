import torch
from torch import nn

if __name__ == "__main__":
    h_d = 768
    pool_dim = 256
    
    dim_reduce = nn.Linear(h_d, pool_dim).cuda()
    avg_pool = nn.Sequential()
    avg_pool.add_module('drop', nn.Dropout(p=0.5))
    avg_pool.add_module('adaptive_avg_pool_1d', nn.AdaptiveAvgPool1d(pool_dim))

    loss_fn = nn.NLLLoss()

    x = torch.randn([1, 5000, h_d]).cuda()
    y = torch.Tensor([1]).long().cuda()

    x_1 = dim_reduce(x)
    x_2 = avg_pool(x_1.permute(0, 2, 1))
    output = x_2.reshape(1, -1)

    loss = loss_fn(output, y)
    loss.backward()
