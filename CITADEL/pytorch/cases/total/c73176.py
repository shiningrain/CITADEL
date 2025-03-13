
results = dict()
import torch
torch.manual_seed(0)
arg_class = torch.nn.MultiLabelMarginLoss()
input_tensor = torch.rand([1, 4], dtype=torch.float32)
target_tensor = torch.randint(256,16384,[1, 4], dtype=torch.int64)

try:
  results["res_cpu"] = arg_class(input_tensor.clone(), target_tensor.clone())
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)

arg_class = arg_class.cuda()
try:
  results["res_gpu"] = arg_class(input_tensor.clone().cuda(), target_tensor.clone().cuda())
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(str(results))
# {'err_cpu': "ERROR:argument #2 'target' is out of range", 'res_gpu': tensor(6.9519, device='cuda:0')}

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)