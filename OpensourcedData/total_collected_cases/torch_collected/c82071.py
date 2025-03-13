
import torch
torch.random.manual_seed(420)
input = torch.randn(2,3,requires_grad=True)
res_cpu = torch.nn.functional.gumbel_softmax(input, hard=True).detach().numpy()
# print("res_cpu: ", res_cpu)
input2 = input.clone().detach().to('cuda')
res_gpu = torch.nn.functional.gumbel_softmax(input2, hard=True).cpu().detach().numpy()
print(res_gpu-res_cpu)

save_path='./tmp_result.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(r_e_s, f)