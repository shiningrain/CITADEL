
import torch
torch.random.manual_seed(420)
input = torch.randn(3, 3, dtype=torch.float32)
print("Intermediate: ", torch.log(input * 2 - 1)) # Contains nan
output = torch.matrix_exp(torch.log(input * 2 - 1))
print("cpu output: ", output)
input = input.cuda()
output = torch.matrix_exp(torch.log(input * 2 - 1))
print("gpu output: ", output) 
