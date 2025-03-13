
import torch
torch.manual_seed(0)
results={}
arg_1 = torch.rand([1, 64, 8], dtype=torch.float32)
arg_2 = -255
arg_3 = False
results['res'] = torch.nn.functional.adaptive_max_pool1d(arg_1,arg_2,arg_3,)
print(results['res'].shape[-1])
#torch.Size([1, 64, -255])

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)