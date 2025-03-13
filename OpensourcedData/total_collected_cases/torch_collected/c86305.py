
import torch
input = torch.tensor([[1.0]*2, [3]*2], requires_grad=True).cuda() 
output = torch.linalg.inv(torch.matmul(input, input))
print(output)
