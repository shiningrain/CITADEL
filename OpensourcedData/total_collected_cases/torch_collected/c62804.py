
import torch
from torch import nn
import time

torch.manual_seed(0)

# Define parameters
M, N, K = 1, 64, 32 # 156800, 16, 36

x = 2*torch.randn((M, K), dtype=torch.float)+2
x_scale = 1.2
x_zero_point = 0

w = 2*torch.randn((N, K), dtype=torch.float)
w_scale = 0.2
w_zero_point = 0

bias_float = 3*torch.randn(N, dtype=torch.float)+1

y_scale = 4.2
y_zero_point = 0

# Prepack weight
def prepack_weight(qw):
    return torch.ops.quantized.linear_prepack(qw, bias_float)

# Do qlinear
def linear(qx, w_packed):
    return torch.ops.quantized.linear(qx, w_packed, y_scale, y_zero_point)

run_round = 100

##### int8 #####
# Specify qengine
torch.backends.quantized.engine = 'fbgemm'
# Quantize x, w and prepack w
qx = torch.quantize_per_tensor(x, scale=x_scale, zero_point=x_zero_point, dtype=torch.quint8)
qw = torch.quantize_per_tensor(w, scale=w_scale, zero_point=w_zero_point, dtype=torch.qint8)
w_packed = prepack_weight(qw)

# do qlinear and profiling
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    t0=time.time()
    for i in range(run_round):
        
        linear_ret = linear(qx, w_packed)
t1=time.time()-t0
# prof.export_chrome_trace('fbgemm-int8-linear.json')
# table_res = prof.key_averages().table(sort_by="cpu_time_total")
# print('----- FBGEMM int8 qlinear -----')
# print(table_res)


##### fp32 #####
m = nn.Linear(K, N)
input = torch.randn(M, K)

with torch.autograd.profiler.profile(use_cuda=False) as prof:
    t0=time.time()
    for i in range(run_round):
        out = m(input)
t2=time.time()-t0

print((t1-t2)/t2)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)