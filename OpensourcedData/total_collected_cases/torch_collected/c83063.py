
import torch
torch.manual_seed(0)
results={}
arg_1 = -7
arg_class = torch.nn.AdaptiveMaxPool3d(arg_1,)
arg_2_0_tensor = torch.rand([1, 64, 10, 9, 8], dtype=torch.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
results['res'] = arg_class(*arg_2)
print(results['res'].shape[-1])
#torch.Size([1, 64, -7, -7, -7])

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)
