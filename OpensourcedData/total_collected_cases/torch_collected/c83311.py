
results = dict()
import torch
torch.manual_seed(0)
arg_1 = torch.rand([3, 3, 5, 5], dtype=torch.complex128)
try:
  results["res_cpu"] = torch.cholesky_inverse(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_1.clone().cuda()
try:
  results["res_gpu"] = torch.cholesky_inverse(arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(str(results))


save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)