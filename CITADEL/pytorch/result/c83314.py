
results = dict()
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
torch.manual_seed(0)
arg_1 = torch.randint(-32768,8,[2], dtype=torch.int64)
arg_2 = torch.randint(-1,32,[4], dtype=torch.int8)
try:
  results["res_cpu"] = func_cls(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1.clone().cuda()
arg_2 = arg_2.clone().cuda()
try:
  results["res_gpu"] = func_cls(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
  
r_e_s=str(results)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)