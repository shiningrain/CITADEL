
results = dict()
import torch
torch.manual_seed(0)
arg_1 = torch.randint(-32768,8,[2], dtype=torch.int64)
arg_2 = torch.randint(-1,32,[4], dtype=torch.int8)
try:
  results["res_cpu"] = torch.equal(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1.clone().cuda()
arg_2 = arg_2.clone().cuda()
try:
  results["res_gpu"] = torch.equal(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
  
print(str(results))

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)