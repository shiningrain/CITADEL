
import torch
import time
torch.manual_seed(0)
batch_size = 100
tgt_dim = 1000000
src_dim = 300000
max_spread_pow = 10
fp32_results = []
fp16_results = []

spread_pow=max_spread_pow
for precision in ["fp32", "fp16"]:
    dtype = torch.float32 if precision == "fp32" else torch.float16
    tgt = torch.zeros(batch_size, tgt_dim).to(dtype).cuda()
    src = torch.rand(batch_size, src_dim).to(dtype).cuda()
    spread = 0.5 ** spread_pow
    index = torch.randint(
        int(tgt_dim / 2 - tgt_dim * spread),
        int(tgt_dim / 2 + tgt_dim * spread),
        (batch_size, src_dim)
    ).cuda()

    t_mean = 0
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.scatter_add(tgt,1, index, src)#randint,gather,index_add|scatter_add
        torch.cuda.synchronize()
        t1 = time.time()
        t_mean += t1 - t0
    t_mean = t_mean / 100

    if precision == "fp32":
        fp32_results.append(t_mean)
    else:
        fp16_results.append(t_mean)

print((fp16_results[0]-fp32_results[0])/fp32_results[0])

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)