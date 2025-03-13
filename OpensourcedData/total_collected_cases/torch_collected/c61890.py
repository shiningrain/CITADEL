
import torch

device = torch.device("cuda:0")
cur_mat = torch.eye(3).to(device).unsqueeze(0)
cur_vec = torch.as_tensor([-0.6660, -0.2958,  8.6392]).view(3, 1).to(device).unsqueeze(0)

print("Mat")
print(cur_mat)
print("Vec")
print(cur_vec)
print("----------")
print("CPU: torch.bmm   : ", end="")
print(cur_mat.cpu().bmm(cur_vec.cpu()).view(3))
print("GPU: torch.bmm   : ", end="")
print(cur_mat.bmm(cur_vec).view(3))
print("GPU: torch.matmul: ", end="")
print(cur_mat.matmul(cur_vec).view(3))
print("GPU: torch.mm    : ", end="")
print(cur_mat[0].mm(cur_vec[0]).view(3))
