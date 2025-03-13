
import torch 

input_tensor = torch.ones((1,1,512,512))

sparse = torch.sparse_coo_tensor(size=(1,10,512,512))

dense  = torch.zeros((1,10,512,512))

print(torch.mul(input_tensor,dense).size())

print(torch.mul(input_tensor,sparse).size())
